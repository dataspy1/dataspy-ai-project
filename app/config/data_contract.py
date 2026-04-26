DATA_CONTRACT = {
    "required_roles": {
        "date": ["date", "order_date", "transaction_date"],
        "product": ["product", "item", "product_name"],
        "revenue": ["revenue", "sales", "amount", "total"],
        "quantity": ["quantity", "units", "qty"],
        "region": ["region", "city", "state", "location"],
    },

    "optional_roles": {
        "order_id": ["order_id"],
        "delivery_date": ["delivery_date"],
        "shipment_date": ["shipment_date"],
        "status": ["status"],
        "stock": ["stock", "inventory"],
        "reorder_level": ["reorder_level"],
        "lead_time": ["lead_time"],
    },

    "data_rules": {
        "date_format": "YYYY-MM-DD",
        "numeric_fields": ["revenue", "quantity"],
        "no_null_columns": ["date", "revenue", "product"],
    },

    "minimum_requirements": {
        "rows_for_insights": 100,
        "rows_for_forecast": 500,
        "time_series_required": True
    }
}