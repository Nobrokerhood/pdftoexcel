from dataclasses import dataclass


MEMBER_RECEIPT = "MEMBER_RECEIPT"
VENDOR_INVOICE = "VENDOR_INVOICE"


@dataclass(frozen=True)
class PurposeDefinition:
    code: str
    label: str
    enabled: bool = True


PURPOSES = [
    PurposeDefinition(MEMBER_RECEIPT, "Member Bank Receipt"),
    PurposeDefinition(VENDOR_INVOICE, "Vendor Invoice"),
]


def supported_purpose_codes() -> set[str]:
    return {purpose.code for purpose in PURPOSES if purpose.enabled}
