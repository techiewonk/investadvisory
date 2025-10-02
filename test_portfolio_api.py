"""
Simple test script for portfolio APIs.
Run this after setting up the database with seed_portfolio_db.py
"""

import asyncio
import json
from decimal import Decimal

import httpx


class DecimalEncoder(json.JSONEncoder):
    """JSON encoder that handles Decimal objects."""
    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)
        return super().default(obj)


async def test_portfolio_apis():
    """Test the portfolio API endpoints."""
    
    base_url = "http://localhost:8080"  # Adjust if your service runs on different port
    
    async with httpx.AsyncClient() as client:
        print(f"\n{'='*50}")
        print("Testing Get All Clients API")
        print(f"{'='*50}")
        
        # Test GET all clients
        try:
            response = await client.get(f"{base_url}/portfolio/clients")
            if response.status_code == 200:
                data = response.json()
                print("✅ All clients:")
                print(json.dumps(data, indent=2, cls=DecimalEncoder))
                
                # Extract client IDs for further testing
                test_client_ids = [c["client_id"] for c in data.get("clients", [])]
                print(f"\n📋 Found {len(test_client_ids)} clients: {test_client_ids}")
            else:
                print(f"❌ Error getting all clients: {response.status_code} - {response.text}")
                # Fallback to hardcoded client IDs
                test_client_ids = ["CLT-001", "CLT-002", "CLT-003"]
        except Exception as e:
            print(f"❌ Exception getting all clients: {e}")
            # Fallback to hardcoded client IDs
            test_client_ids = ["CLT-001", "CLT-002", "CLT-003"]
        
        # Now test individual client endpoints
        for client_id in test_client_ids:
            print(f"\n{'='*50}")
            print(f"Testing Client ID: {client_id}")
            print(f"{'='*50}")
            
            # Test GET portfolios
            try:
                response = await client.get(f"{base_url}/portfolio/users/{client_id}/portfolios")
                if response.status_code == 200:
                    data = response.json()
                    print(f"✅ Portfolios for {client_id}:")
                    print(json.dumps(data, indent=2, cls=DecimalEncoder))
                else:
                    print(f"❌ Error getting portfolios: {response.status_code} - {response.text}")
            except Exception as e:
                print(f"❌ Exception getting portfolios: {e}")
            
            # Test GET transactions
            try:
                response = await client.get(f"{base_url}/portfolio/users/{client_id}/transactions?limit=5")
                if response.status_code == 200:
                    data = response.json()
                    print(f"\n✅ Recent transactions for {client_id}:")
                    print(json.dumps(data, indent=2, cls=DecimalEncoder))
                else:
                    print(f"❌ Error getting transactions: {response.status_code} - {response.text}")
            except Exception as e:
                print(f"❌ Exception getting transactions: {e}")
            
            # Test POST portfolios
            try:
                payload = {"client_id": client_id}
                response = await client.post(f"{base_url}/portfolio/user-portfolios", json=payload)
                if response.status_code == 200:
                    print(f"\n✅ POST portfolios for {client_id}: Success")
                else:
                    print(f"❌ Error POST portfolios: {response.status_code} - {response.text}")
            except Exception as e:
                print(f"❌ Exception POST portfolios: {e}")


if __name__ == "__main__":
    print("🚀 Testing Portfolio APIs")
    print("Make sure your service is running on http://localhost:8080")
    print("And that you've seeded the database with seed_portfolio_db.py")
    
    asyncio.run(test_portfolio_apis())
