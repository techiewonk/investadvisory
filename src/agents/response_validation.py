"""Response Validation System for Investment Advisory Platform.

This module provides comprehensive validation to detect hallucinations and ensure response reliability.
"""

import json
import logging
import re
from datetime import datetime
from functools import wraps
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import yfinance as yf
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ValidationResult(BaseModel):
    """Result of response validation."""
    
    is_valid: bool = Field(description="Whether the response passed validation")
    confidence_score: float = Field(ge=0.0, le=1.0, description="Confidence score (0-1)")
    validation_checks: Dict[str, bool] = Field(description="Individual validation check results")
    errors: List[str] = Field(default_factory=list, description="Validation errors found")
    warnings: List[str] = Field(default_factory=list, description="Validation warnings")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional validation metadata")


class ResponseValidator:
    """Comprehensive response validator for investment advisory responses."""
    
    def __init__(self):
        self.validation_cache = {}
        self.confidence_thresholds = {
            'high': 0.9,
            'medium': 0.7,
            'low': 0.5
        }
    
    def validate_response(self, response: Any, context: Dict[str, Any] = None) -> ValidationResult:
        """
        Validate a response for accuracy and reliability.
        
        Args:
            response: The response to validate
            context: Additional context for validation
            
        Returns:
            ValidationResult with validation status and details
        """
        if context is None:
            context = {}
        
        validation_checks = {}
        errors = []
        warnings = []
        metadata = {}
        
        # Convert response to string for analysis
        response_text = str(response) if not isinstance(response, str) else response
        
        # 1. Data Source Validation
        data_validation = self._validate_data_sources(response_text, context)
        validation_checks['data_sources'] = data_validation['valid']
        errors.extend(data_validation['errors'])
        warnings.extend(data_validation['warnings'])
        
        # 2. Numerical Validation
        numerical_validation = self._validate_numerical_data(response_text, context)
        validation_checks['numerical_data'] = numerical_validation['valid']
        errors.extend(numerical_validation['errors'])
        warnings.extend(numerical_validation['warnings'])
        
        # 3. Financial Logic Validation
        logic_validation = self._validate_financial_logic(response_text, context)
        validation_checks['financial_logic'] = logic_validation['valid']
        errors.extend(logic_validation['errors'])
        warnings.extend(logic_validation['warnings'])
        
        # 4. Consistency Validation
        consistency_validation = self._validate_consistency(response_text, context)
        validation_checks['consistency'] = consistency_validation['valid']
        errors.extend(consistency_validation['errors'])
        warnings.extend(consistency_validation['warnings'])
        
        # 5. Market Data Validation
        if 'symbols' in context or self._contains_stock_symbols(response_text):
            market_validation = self._validate_market_data(response_text, context)
            validation_checks['market_data'] = market_validation['valid']
            errors.extend(market_validation['errors'])
            warnings.extend(market_validation['warnings'])
        
        # Calculate overall confidence score
        confidence_score = self._calculate_confidence_score(validation_checks, errors, warnings)
        
        # Determine overall validity
        is_valid = confidence_score >= self.confidence_thresholds['low'] and len(errors) == 0
        
        metadata.update({
            'validation_timestamp': datetime.now().isoformat(),
            'confidence_level': self._get_confidence_level(confidence_score),
            'total_checks': len(validation_checks),
            'passed_checks': sum(validation_checks.values()),
            'failed_checks': len(validation_checks) - sum(validation_checks.values())
        })
        
        return ValidationResult(
            is_valid=is_valid,
            confidence_score=confidence_score,
            validation_checks=validation_checks,
            errors=errors,
            warnings=warnings,
            metadata=metadata
        )
    
    def _validate_data_sources(self, response: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate that response data comes from reliable sources."""
        errors = []
        warnings = []
        valid = True
        
        # Check for unrealistic claims
        unrealistic_patterns = [
            r'guaranteed.*returns?',
            r'risk-free.*investment',
            r'100%.*profit',
            r'never.*lose.*money',
            r'always.*profitable'
        ]
        
        for pattern in unrealistic_patterns:
            if re.search(pattern, response, re.IGNORECASE):
                errors.append(f"Unrealistic claim detected: {pattern}")
                valid = False
        
        # Check for proper disclaimers on predictions
        prediction_patterns = [
            r'will.*increase.*\d+%',
            r'expect.*returns?.*\d+%',
            r'predict.*price.*\$\d+'
        ]
        
        has_disclaimer = any(disclaimer in response.lower() for disclaimer in [
            'past performance', 'not guarantee', 'may vary', 'risk involved', 'consult'
        ])
        
        for pattern in prediction_patterns:
            if re.search(pattern, response, re.IGNORECASE) and not has_disclaimer:
                warnings.append("Prediction made without appropriate disclaimers")
        
        return {'valid': valid, 'errors': errors, 'warnings': warnings}
    
    def _validate_numerical_data(self, response: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate numerical data for plausibility."""
        errors = []
        warnings = []
        valid = True
        
        # Extract numerical values
        percentages = re.findall(r'(\d+(?:\.\d+)?)\s*%', response)
        dollar_amounts = re.findall(r'\$(\d+(?:,\d{3})*(?:\.\d{2})?)', response)
        ratios = re.findall(r'(\d+(?:\.\d+)?):1', response)
        
        # Validate percentages
        for pct in percentages:
            pct_val = float(pct)
            if pct_val > 1000:  # Unrealistic percentage
                errors.append(f"Unrealistic percentage value: {pct}%")
                valid = False
            elif pct_val > 100 and 'return' in response.lower():
                warnings.append(f"High return percentage: {pct}%")
        
        # Validate dollar amounts
        for amount in dollar_amounts:
            amount_val = float(amount.replace(',', ''))
            if amount_val > 1e12:  # Trillion dollars - unrealistic for individual portfolios
                errors.append(f"Unrealistic dollar amount: ${amount}")
                valid = False
        
        # Validate ratios
        for ratio in ratios:
            ratio_val = float(ratio)
            if ratio_val > 100:  # Extremely high ratio
                warnings.append(f"High ratio value: {ratio}:1")
        
        return {'valid': valid, 'errors': errors, 'warnings': warnings}
    
    def _validate_financial_logic(self, response: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate financial logic and relationships."""
        errors = []
        warnings = []
        valid = True
        
        # Check for logical inconsistencies
        if 'low risk' in response.lower() and 'high return' in response.lower():
            if not any(qualifier in response.lower() for qualifier in ['historically', 'typically', 'generally']):
                warnings.append("Low risk + high return claim without proper qualification")
        
        # Validate P/E ratios
        pe_matches = re.findall(r'P/E.*?(\d+(?:\.\d+)?)', response, re.IGNORECASE)
        for pe in pe_matches:
            pe_val = float(pe)
            if pe_val < 0:
                errors.append(f"Negative P/E ratio: {pe}")
                valid = False
            elif pe_val > 1000:
                errors.append(f"Unrealistic P/E ratio: {pe}")
                valid = False
            elif pe_val > 100:
                warnings.append(f"Very high P/E ratio: {pe}")
        
        # Validate RSI values
        rsi_matches = re.findall(r'RSI.*?(\d+(?:\.\d+)?)', response, re.IGNORECASE)
        for rsi in rsi_matches:
            rsi_val = float(rsi)
            if rsi_val < 0 or rsi_val > 100:
                errors.append(f"Invalid RSI value (must be 0-100): {rsi}")
                valid = False
        
        return {'valid': valid, 'errors': errors, 'warnings': warnings}
    
    def _validate_consistency(self, response: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate internal consistency of the response."""
        errors = []
        warnings = []
        valid = True
        
        # Check for contradictory statements
        contradictions = [
            (['bullish', 'positive'], ['bearish', 'negative']),
            (['buy', 'accumulate'], ['sell', 'reduce']),
            (['overvalued'], ['undervalued']),
            (['low risk'], ['high risk'])
        ]
        
        response_lower = response.lower()
        for positive_terms, negative_terms in contradictions:
            has_positive = any(term in response_lower for term in positive_terms)
            has_negative = any(term in response_lower for term in negative_terms)
            
            if has_positive and has_negative:
                warnings.append(f"Potentially contradictory statements detected")
        
        return {'valid': valid, 'errors': errors, 'warnings': warnings}
    
    def _validate_market_data(self, response: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate market data against real sources."""
        errors = []
        warnings = []
        valid = True
        
        # Extract stock symbols
        symbols = self._extract_stock_symbols(response)
        
        if not symbols:
            return {'valid': valid, 'errors': errors, 'warnings': warnings}
        
        # Validate up to 3 symbols to avoid rate limits
        for symbol in symbols[:3]:
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                
                if not info or 'symbol' not in info:
                    warnings.append(f"Could not validate market data for {symbol}")
                    continue
                
                # Extract prices mentioned in response for this symbol
                price_pattern = rf'{symbol}.*?\$(\d+(?:\.\d{{2}})?)'
                price_matches = re.findall(price_pattern, response, re.IGNORECASE)
                
                if price_matches:
                    mentioned_price = float(price_matches[0])
                    current_price = info.get('currentPrice') or info.get('regularMarketPrice', 0)
                    
                    if current_price > 0:
                        price_diff_pct = abs(mentioned_price - current_price) / current_price * 100
                        
                        if price_diff_pct > 50:  # More than 50% difference
                            errors.append(f"Price for {symbol} (${mentioned_price}) differs significantly from current market price (${current_price})")
                            valid = False
                        elif price_diff_pct > 20:  # More than 20% difference
                            warnings.append(f"Price for {symbol} may be outdated")
                
            except Exception as e:
                logger.warning(f"Could not validate market data for {symbol}: {e}")
        
        return {'valid': valid, 'errors': errors, 'warnings': warnings}
    
    def _contains_stock_symbols(self, text: str) -> bool:
        """Check if text contains stock symbols."""
        # Common stock symbol patterns
        symbol_patterns = [
            r'\b[A-Z]{1,5}\b',  # 1-5 letter uppercase
            r'\$[A-Z]{1,5}\b',  # With dollar sign prefix
        ]
        
        for pattern in symbol_patterns:
            if re.search(pattern, text):
                return True
        return False
    
    def _extract_stock_symbols(self, text: str) -> List[str]:
        """Extract stock symbols from text."""
        # Common stock symbols (1-5 letters, uppercase)
        symbols = re.findall(r'\b([A-Z]{1,5})\b', text)
        
        # Filter out common words that might match the pattern
        common_words = {'THE', 'AND', 'FOR', 'ARE', 'BUT', 'NOT', 'YOU', 'ALL', 'CAN', 'HER', 'WAS', 'ONE', 'OUR', 'HAD', 'BY', 'UP', 'DO', 'NO', 'IF', 'MY', 'I', 'IT', 'AM', 'IS', 'HE', 'AS', 'TO', 'WE', 'OR', 'AN', 'US', 'AT', 'ON', 'BE', 'GO', 'SO', 'IN', 'OF', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'}
        
        filtered_symbols = [s for s in symbols if s not in common_words]
        return list(set(filtered_symbols))  # Remove duplicates
    
    def _calculate_confidence_score(self, validation_checks: Dict[str, bool], errors: List[str], warnings: List[str]) -> float:
        """Calculate overall confidence score."""
        if not validation_checks:
            return 0.5  # Neutral confidence if no checks
        
        # Base score from passed validation checks
        passed_checks = sum(validation_checks.values())
        total_checks = len(validation_checks)
        base_score = passed_checks / total_checks
        
        # Penalties for errors and warnings
        error_penalty = len(errors) * 0.2  # 20% penalty per error
        warning_penalty = len(warnings) * 0.05  # 5% penalty per warning
        
        # Calculate final score
        confidence_score = max(0.0, base_score - error_penalty - warning_penalty)
        return min(1.0, confidence_score)
    
    def _get_confidence_level(self, score: float) -> str:
        """Get confidence level from score."""
        if score >= self.confidence_thresholds['high']:
            return 'high'
        elif score >= self.confidence_thresholds['medium']:
            return 'medium'
        elif score >= self.confidence_thresholds['low']:
            return 'low'
        else:
            return 'very_low'


# Global validator instance
response_validator = ResponseValidator()


def validate_response(context: Optional[Dict[str, Any]] = None):
    """
    Decorator to validate tool responses.
    
    Args:
        context: Additional context for validation
    """
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                # Execute the original function
                result = await func(*args, **kwargs)
                
                # Validate the result
                validation_context = context or {}
                validation_context.update({
                    'function_name': func.__name__,
                    'args': args,
                    'kwargs': kwargs
                })
                
                validation_result = response_validator.validate_response(result, validation_context)
                
                # If validation fails, modify the response
                if not validation_result.is_valid:
                    logger.warning(f"Validation failed for {func.__name__}: {validation_result.errors}")
                    
                    # Add validation information to response
                    if isinstance(result, dict):
                        result['validation'] = {
                            'status': 'failed',
                            'confidence_score': validation_result.confidence_score,
                            'errors': validation_result.errors,
                            'warnings': validation_result.warnings
                        }
                    else:
                        # For non-dict responses, create a new dict
                        result = {
                            'original_response': result,
                            'validation': {
                                'status': 'failed',
                                'confidence_score': validation_result.confidence_score,
                                'errors': validation_result.errors,
                                'warnings': validation_result.warnings
                            }
                        }
                
                elif validation_result.warnings:
                    logger.info(f"Validation warnings for {func.__name__}: {validation_result.warnings}")
                    
                    # Add warnings to response
                    if isinstance(result, dict):
                        result['validation'] = {
                            'status': 'passed_with_warnings',
                            'confidence_score': validation_result.confidence_score,
                            'warnings': validation_result.warnings
                        }
                
                return result
                
            except Exception as e:
                logger.error(f"Error in validation wrapper for {func.__name__}: {e}")
                return await func(*args, **kwargs)  # Return original result if validation fails
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                # Execute the original function
                result = func(*args, **kwargs)
                
                # Validate the result
                validation_context = context or {}
                validation_context.update({
                    'function_name': func.__name__,
                    'args': args,
                    'kwargs': kwargs
                })
                
                validation_result = response_validator.validate_response(result, validation_context)
                
                # If validation fails, modify the response
                if not validation_result.is_valid:
                    logger.warning(f"Validation failed for {func.__name__}: {validation_result.errors}")
                    
                    # Add validation information to response
                    if isinstance(result, dict):
                        result['validation'] = {
                            'status': 'failed',
                            'confidence_score': validation_result.confidence_score,
                            'errors': validation_result.errors,
                            'warnings': validation_result.warnings
                        }
                    else:
                        # For non-dict responses, create a new dict
                        result = {
                            'original_response': result,
                            'validation': {
                                'status': 'failed',
                                'confidence_score': validation_result.confidence_score,
                                'errors': validation_result.errors,
                                'warnings': validation_result.warnings
                            }
                        }
                
                elif validation_result.warnings:
                    logger.info(f"Validation warnings for {func.__name__}: {validation_result.warnings}")
                    
                    # Add warnings to response
                    if isinstance(result, dict):
                        result['validation'] = {
                            'status': 'passed_with_warnings',
                            'confidence_score': validation_result.confidence_score,
                            'warnings': validation_result.warnings
                        }
                
                return result
                
            except Exception as e:
                logger.error(f"Error in validation wrapper for {func.__name__}: {e}")
                return func(*args, **kwargs)  # Return original result if validation fails
        
        # Return appropriate wrapper based on function type
        if hasattr(func, '__code__') and 'async' in str(func.__code__.co_flags):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


# Confidence-based response filtering
def filter_low_confidence_responses(response: Any, threshold: float = 0.5) -> Tuple[Any, bool]:
    """
    Filter responses based on confidence score.
    
    Args:
        response: Response to filter
        threshold: Minimum confidence threshold
        
    Returns:
        Tuple of (filtered_response, should_show)
    """
    if isinstance(response, dict) and 'validation' in response:
        validation_info = response['validation']
        confidence_score = validation_info.get('confidence_score', 1.0)
        
        if confidence_score < threshold:
            return {
                'message': "I'm not confident in this response. Please try rephrasing your question or ask for clarification.",
                'confidence_score': confidence_score,
                'original_available': True
            }, False
    
    return response, True
