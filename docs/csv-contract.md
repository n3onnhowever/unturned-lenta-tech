# CSV Contract

The CSV export is the source-of-truth delivery format for the hackathon task.

## Columns

```text
filename
product_name
price_default
price_card
price_discount
barcode
discount_amount
id_sku
print_datetime
code
additional_info
color
special_symbols
frame_timestamp
x_min
y_min
x_max
y_max
qr_code_barcode
price1_qr
price2_qr
price3_qr
price4_qr
wholesale_level_1_count
wholesale_level_1_price
wholesale_level_2_count
wholesale_level_2_price
action_price_qr
action_code_qr
```

## Rules

- One row equals one unique detected price tag.
- If a field is not present on a price tag, use `нет`.
- If a field is present but was not recognized, leave it empty.
- Encoding: UTF-8.
- Delimiter: comma.

## Example row shape

```text
filename,product_name,price_default,price_card,...,action_code_qr
robot_scan.mp4,Молоко 2.5%,89.99,79.99,...,нет
```
