"""
Vendor Invoice Accounting Engine.
Processes vendor invoices, labour bills, cash memos, and tax invoices.
Extracts vendor details, GSTIN, line items, taxable values, CGST/SGST/IGST breakdown.
Ensures tax lines and subtotal lines are never confused with the total payable voucher amount.
"""

from decimal import Decimal, InvalidOperation
from typing import Dict, Any, List, Optional

from app.accounting.semantic_rows import classify_semantic_row, RowType
from app.accounting.intermediate import IntermediateTransaction, IntermediateDocument


class VendorInvoiceEngine:
    """Deterministic domain logic for Vendor Invoices and Bills."""

    def process(self, raw_data: Dict[str, Any]) -> IntermediateDocument:
        doc = IntermediateDocument(
            purpose="VENDOR_INVOICE",
            document_type=raw_data.get("document_type", "TAX_INVOICE"),
            accounting_content="LINE_ITEM_INVOICE",
            metadata=raw_data.get("metadata", {})
        )

        vendor = str(raw_data.get("vendor_name") or raw_data.get("vendor") or "-").strip()
        invoice_no = str(raw_data.get("invoice_number") or raw_data.get("bill_number") or "-").strip()
        invoice_date = str(raw_data.get("invoice_date") or raw_data.get("bill_date") or "-").strip()
        total_amt_raw = raw_data.get("total_amount") or raw_data.get("invoice_total")

        if total_amt_raw not in {None, "", "-"}:
            try:
                doc.total_expenditure = Decimal(str(total_amt_raw).replace(",", "").strip())
            except (InvalidOperation, ValueError):
                pass

        raw_items = raw_data.get("items") or raw_data.get("rows") or raw_data.get("expenses") or []
        
        # If line items are provided
        if raw_items:
            for idx, item in enumerate(raw_items, start=1):
                desc = str(item.get("description") or item.get("expense_description") or item.get("Particulars") or item.get("Comments") or "-").strip()
                amt_raw = item.get("amount") or item.get("expense_amount") or item.get("Amount*") or item.get("total")
                amt = Decimal("0")
                if amt_raw not in {None, "", "-"}:
                    try:
                        amt = Decimal(str(amt_raw).replace(",", "").strip())
                    except (InvalidOperation, ValueError):
                        amt = Decimal("0")

                row_type = classify_semantic_row(desc, amount=amt, document_purpose="VENDOR_INVOICE")
                
                # Check tax rows: CGST, SGST, IGST
                desc_lower = desc.lower()
                if any(tax in desc_lower for tax in ["cgst", "sgst", "igst"]):
                    row_type = RowType.SUBTOTAL

                comment = f"{vendor} - {desc}" if vendor != "-" else desc

                tx = IntermediateTransaction(
                    transaction_id=f"vend_{idx}",
                    date=invoice_date,
                    description=desc,
                    amount=amt,
                    reference=invoice_no,
                    row_type=row_type,
                    page=int(item.get("page", 1)),
                    source_row_idx=idx,
                    payment_type=str(item.get("Payment Type*") or "Cash").strip(),
                    bill_head=str(item.get("Bill Head*") or "Vendor Payment").strip(),
                    comments=comment,
                    user_edited=bool(item.get("USER_EDITED", False)),
                )

                if row_type in [RowType.TOTAL, RowType.SUBTOTAL]:
                    doc.non_transaction_rows.append(tx)
                else:
                    doc.transactions.append(tx)

            if not doc.total_expenditure and doc.transactions:
                doc.total_expenditure = sum(t.amount for t in doc.transactions if t.row_type in {RowType.GENUINE_TRANSACTION, RowType.EXPENSE})

        # If no item rows exist, but document total is present, create one primary transaction
        if not doc.transactions and doc.total_expenditure:
            tx = IntermediateTransaction(
                transaction_id="vend_primary_1",
                date=invoice_date,
                description=f"Payment to {vendor}",
                amount=doc.total_expenditure,
                reference=invoice_no,
                row_type=RowType.GENUINE_TRANSACTION,
                page=1,
                source_row_idx=1,
                payment_type="Cash",
                bill_head="Vendor Payment",
                comments=f"Bill No: {invoice_no} Vendor: {vendor}"
            )
            doc.transactions.append(tx)

        return doc
