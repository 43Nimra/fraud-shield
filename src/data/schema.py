import sys
sys.path.insert(0, '.')

from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime


class ValidationError(Exception):
    """Raised when transaction data invalid ho"""
    pass


@dataclass
class FraudTransaction:
    """
    Ek transaction ka complete schema.
    WHY dataclass: dict se safer — typos runtime pe nahi,
    type hints se IDE help karta hai
    """
    transaction_id: str
    amount: float
    product_type: str
    card_type: Optional[str] = None
    email_domain: Optional[str] = None
    is_fraud: Optional[int] = None
    created_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        """Validation — object bante waqt chalti hai"""
        if self.amount < 0:
            raise ValidationError(
                f"Amount negative nahi ho sakta: {self.amount}"
            )
        if self.amount > 50000:
            raise ValidationError(
                f"Amount suspiciously high: {self.amount}"
            )
        if not self.transaction_id.strip():
            raise ValidationError("Transaction ID empty nahi ho sakta")
        # amount round karo 2 decimal places
        self.amount = round(self.amount, 2)

    @property
    def is_high_value(self) -> bool:
        """$500 se upar = high value transaction"""
        return self.amount > 500

    @property
    def risk_level(self) -> str:
        """Amount ke basis pe risk level"""
        if self.amount < 50:
            return "LOW"
        elif self.amount < 500:
            return "MEDIUM"
        else:
            return "HIGH"

    def to_dict(self) -> dict:
        """API response ke liye dict conversion"""
        return {
            "transaction_id": self.transaction_id,
            "amount": self.amount,
            "product_type": self.product_type,
            "risk_level": self.risk_level,
            "is_high_value": self.is_high_value,
            "is_fraud": self.is_fraud
        }


if __name__ == "__main__":
    # Test 1 — valid transaction
    t1 = FraudTransaction(
        transaction_id="TXN001",
        amount=150.50,
        product_type="W",
        email_domain="gmail.com"
    )
    print("Valid transaction:", t1.to_dict())
    print("Risk level:", t1.risk_level)
    print("High value:", t1.is_high_value)

    # Test 2 — high value
    t2 = FraudTransaction(
        transaction_id="TXN002",
        amount=1500.00,
        product_type="C"
    )
    print("\nHigh value transaction:", t2.to_dict())

    # Test 3 — validation error
    print("\nTesting validation...")
    try:
        t3 = FraudTransaction(
            transaction_id="TXN003",
            amount=-100,
            product_type="W"
        )
    except ValidationError as e:
        print(f"ValidationError caught: {e}")

    print("\nSchema OK!")
