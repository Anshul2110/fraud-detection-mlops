from pydantic import BaseModel
from typing import Optional

class TransactionInput(BaseModel):
    TransactionAmt: float
    ProductCD: Optional[str] = "W"
    card1: Optional[float] = 10000
    card2: Optional[float] = 321.0
    card3: Optional[float] = 150.0
    card4: Optional[str] = "visa"
    card5: Optional[float] = 226.0
    card6: Optional[str] = "debit"
    P_emaildomain: Optional[str] = "gmail.com"
    R_emaildomain: Optional[str] = "gmail.com"
    TransactionDT: Optional[int] = 86400

class PredictionResponse(BaseModel):
    transaction_amount: float
    fraud_probability: float
    prediction: str
    risk_level: str
    top_features: dict