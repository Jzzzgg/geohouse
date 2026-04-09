from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
import os
import httpx
import logging
from typing import Dict, Any, Optional
from dotenv import load_dotenv
import asyncio 


load_dotenv()

app = FastAPI(title="GeoHouse",
              description="Enter Address → Get Housing Prices, Crime Rates, Population + AI Analysis",
              version="1.0.0"
              )

app.add_middleware(
    CORSMiddleware,
    allow_origins=["zhuguangjiang.com"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#GET API KEYS
GEOCODIO_API_KEY = os.getenv("GEOCODIO_API_KEY")     
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")
CENSUS_API_KEY = os.getenv("CENSUS_API_KEY")          
AI_API_KEY = os.getenv("GENMINI_API_KEY")
ZILLOW_API_KEY = os.getenv("ZILLOW_API_KEY")
CRIME_API_KEY = os.getenv("CRIME_API_KEY")


class AddressRequest(BaseModel):
    address: str
    language: Optional[str] = "zh"   #language to return

class AnalysisResponse(BaseModel):
    address: str
    coordinates: Dict[str, float]
    housing: Dict[str, Any]
    crime: Dict[str, Any]
    population: Dict[str, Any]
    ai_explanation: str
    sources: list[str]


# ====================== functions ======================
async def geocode_address(address: str) -> Dict:
    """Lat + Lon + Census (Geocodio)"""
    if not GEOCODIO_API_KEY:
        raise HTTPException(500, "Geocodio API Key not found")
    
    url = f"https://api.geocod.io/v1/geocode"
    params = {"q": address, "api_key": GEOCODIO_API_KEY, "fields": "census"}
    
    async with httpx.AsyncClient() as client:
        resp = await client.get(url, params=params)
        data = resp.json()
        
        if not data.get("results"):
            raise HTTPException(400, "Unable to analysis address")
        
        result = data["results"][0]
        return {
            "address": result["formatted_address"],
            "lat": result["location"]["lat"],
            "lng": result["location"]["lng"],
            "census": result.get("fields", {}).get("census", {})
        }

async def get_population_data(census_info: Dict) -> Dict:
    """get population info from Census API """
    if not census_info:
        return {"population": "N/A", "density": "N/A"}
    
    census_url = "https://api.census.gov/data/2024/pep/population"
    state_fips = 42
    params = {
        "get": "NAME,POP_2024", 
        "for": f"state:{state_fips}",
        "key": CENSUS_API_KEY
    }

    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(census_url, params=params)
            resp.raise_for_status()
            raw_data = resp.json()

            # handle Census data [["NAME", "POP_2024", "state"], ["California", "39142991", "06"]]
            headers = raw_data[0]
            values = raw_data[1]
            result = dict(zip(headers, values))

            return {
                "state": result["NAME"],
                "population_2024": int(result["POP_2024"]),
                "fips": result["state"]
            }

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Census API error: {str(e)}")
        

async def get_housing_data(lat: float, lng: float, address: str) -> Dict:
    """Zillow api"""

    url = "https://zillow-com1.p.rapidapi.com/property"
    headers = {
        "X-RapidAPI-Key": "ZILLOW_API_KEY",
        "X-RapidAPI-Host": "zillow-com1.p.rapidapi.com"
    }
    
    querystring = {"address": address}
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, headers=headers, params=querystring)
            data = response.json()
            
            price = data.get("price")
            zestimate = data.get("zestimate")
            
            return {"address": address, "price": price, "zestimate": zestimate}
        
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Zillow API error: {str(e)}")
        
async def get_crime_data(lat: float, lng: float) -> Dict:
    """CrimeoMeter api"""

    crime_url = "https://api.crimeometer.com/v2/incidents/stats"
    params = {
        "lat": lat,
        "lon": lng,
        "distance": '2mi',
        "datetime_ini": "2025-01-01T00:00:00Z",
        "datetime_end": "2026-01-01T00:00:00Z"
    }
    
    headers = {"Content-Type": "application/json", "x-api-key": CRIME_API_KEY}

    async with httpx.AsyncClient() as client:
        resp = await client.get(crime_url, headers=headers, params=params)
        data = resp.json()
        
        safety_score = data.get("csi", "N/A")
        total_incidents = data.get("total_incidents", 0)
        
        return {
            "safety_index": safety_score,
            "total_incidents": total_incidents,
            "recommendation": "safe" if safety_score < 40 else "need more attention"
        }

async def generate_ai_explanation(data: Dict, language: str = "zh") -> str:
    """GENMINI AI API """
    if not AI_API_KEY:
        return "No AI API Key"
    
    prompt = f"""
        As a professional Real Estate and Urban Analysis Expert, here is a strategic evaluation based on the provided data for {data['address']}:
        Neighborhood Investment & Livability Profile
        The property value of {data['housing']} suggests this is a significant asset within a high-density submarket. With a population of {data['population']}, the area benefits from "agglomeration effects"—meaning the sheer volume of residents supports diverse local amenities, better transit infrastructure, and a consistent pool of rental demand.
        Strengths & Risk Assessment
        Strengths: The primary advantage here is demographic depth. A large population usually correlates with a resilient local economy and higher "walk scores." In urban analysis, this density often leads to faster appreciation (Equity Growth) as land becomes increasingly scarce.
        Risks: The reported crime level of {data['crime']} is the critical factor to monitor. While urban density naturally increases incident counts, it is essential to analyze whether these are decreasing year-over-year. High crime can cap the "exit price" for future sales, though it often provides a lower entry point for investors looking for "Value-Add" opportunities through gentrification.
        Target Demographic & Suitability
        This area is a prime fit for Strategic Investors and Young Professionals. For investors, the dense population indicates high occupancy rates and strong Cash-on-Cash returns. For young professionals, the urban energy and proximity to employment hubs outweigh the risk profile. While families might prioritize quieter suburban pockets, this location is ideal for those prioritizing urban connectivity and long-term capital gains.
        use { '中文' if language == 'zh' else 'English' } as language， 400 words
    """
    
    client = genai.Client(api_key=AI_API_KEY)

    # 2. use model
    response = client.models.generate_content(
        model="gemini-3-flash-preview",  
        contents=prompt
)
    text_content = response['candidates'][0]['content']['parts'][0]['text']
    return text_content

# ====================== API ======================
@app.get("/")
async def root():
    return {"message": "Geohouse API in processing ！view /docs for Swagger doc"}

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_address(request: AddressRequest):
    try:
        logger.info(f"Analysis address: {request.address}")
        
        # 1. Geocoding
        geo = await geocode_address(request.address)
        
        # 2. get datas
        housing, crime, population = await asyncio.gather(
            get_housing_data(geo["lat"], geo["lng"], request.address),
            get_crime_data(geo["lat"], geo["lng"]),
            get_population_data(geo.get("census", {}))
        )
        
        # 3. AI reading
        full_data = {
            "address": geo["address"],
            "housing": housing,
            "crime": crime,
            "population": population
        }
        ai_text = await generate_ai_explanation(full_data, request.language)
        
        return {
            "address": geo["address"],
            "coordinates": {"lat": geo["lat"], "lng": geo["lng"]},
            "housing": housing,
            "crime": crime,
            "population": population,
            "ai_explanation": ai_text,
            "sources": ["Geocodio", "US Census", "Zillow", "CrimeoMeter"]
        }
        
    except Exception as e:
        logger.error(f"Fail to analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
