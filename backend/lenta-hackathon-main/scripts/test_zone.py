import cv2
from paddleocr import PaddleOCR
import re
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def slice_zones(image):
    h, w = image.shape[:2]
    mid_h = int(h * 0.45)
    
    zone_name = image[0:mid_h, 0:w]
    zone_prices = image[mid_h:h, 0:w]
    
    return zone_name, zone_prices

def extract_text(ocr_result):
    if not ocr_result: return ""
    texts = []
    if isinstance(ocr_result, list) and len(ocr_result) > 0:
        for res in ocr_result:
            if hasattr(res, 'keys') or isinstance(res, dict):
                if 'rec_texts' in res:
                    val = res['rec_texts']
                    if isinstance(val, list): texts.extend(val)
                    elif val: texts.append(str(val))
                elif 'rec_text' in res:
                    val = res['rec_text']
                    if isinstance(val, list): texts.extend(val)
                    elif val: texts.append(str(val))
                elif 'rec_res' in res:
                    rec_res = res['rec_res']
                    if isinstance(rec_res, list):
                        for item in rec_res:
                            if isinstance(item, tuple) or isinstance(item, list):
                                texts.append(item[0])
            elif isinstance(res, list):
                for line in res:
                    if isinstance(line, list) and len(line) == 2:
                        text_info = line[1]
                        if isinstance(text_info, tuple) or isinstance(text_info, list):
                            texts.append(text_info[0])
            elif hasattr(res, 'rec_text'):
                if res.rec_text:
                    if isinstance(res.rec_text, list): texts.extend(res.rec_text)
                    else: texts.append(res.rec_text)
    return " ".join(texts)

def extract_prices_from_bottom(ocr_result):
    if not ocr_result:
        return "", ""
        
    prices = []
    
    if isinstance(ocr_result, list) and len(ocr_result) > 0:
        res = ocr_result[0]
        if isinstance(res, list):
            for line in res:
                if isinstance(line, list) and len(line) == 2:
                    box = line[0]
                    text_info = line[1]
                    if isinstance(text_info, tuple) or isinstance(text_info, list):
                        text = text_info[0]
                        xs = [p[0] for p in box]
                        ys = [p[1] for p in box]
                        cx = sum(xs) / 4.0
                        h = max(ys) - min(ys)
                        
                        clean_text = re.sub(r'[^\d.,]', '', text).replace(',', '.')
                        matches = list(re.finditer(r'(\d+)\.?(\d{0,2})', clean_text))
                        if matches:
                            match = matches[-1]
                            main = match.group(1)
                            cents = match.group(2) if match.group(2) else "00"
                            if len(cents) == 1: cents += "0"
                            price_str = f"{main}.{cents}"
                            try:
                                val = float(price_str)
                                if 0 < val < 10000:
                                    prices.append({
                                        'text': price_str,
                                        'cx': cx,
                                        'h': h,
                                        'val': val
                                    })
                            except ValueError:
                                pass

    if not prices:
        return "", ""
        
    if len(prices) == 1:
        return prices[0]['text'], prices[0]['text']
        
    prices.sort(key=lambda x: x['h'], reverse=True)
    
    price_card = prices[0]['text']
    price_default = prices[1]['text']
    
    if prices[1]['val'] > prices[0]['val']:
        price_default = prices[1]['text']
        price_card = prices[0]['text']
    elif prices[0]['val'] > prices[1]['val']:
        price_card = prices[1]['text']
        price_default = prices[0]['text']
        
    return price_card, price_default

img_path = "runs/eval_e2e_43_15/upscaled/tag_0001_frame_0038.jpg"
image = cv2.imread(img_path)

z_name, z_prices = slice_zones(image)

ocr_std = PaddleOCR(use_textline_orientation=False, lang='ru', use_doc_orientation_classify=False, use_doc_unwarping=False)
ocr_price = PaddleOCR(use_textline_orientation=False, lang='ru', use_doc_orientation_classify=False, use_doc_unwarping=False)

res_name = ocr_std.ocr(z_name)

z_prices_small = cv2.resize(z_prices, (0, 0), fx=0.5, fy=0.5)
res_prices = ocr_price.ocr(z_prices_small)

print("--- RAW OCR RESULTS ---")
print("Name:", extract_text(res_name))
print("Prices:", extract_prices_from_bottom(res_prices))
print("Raw res_prices:", res_prices)
