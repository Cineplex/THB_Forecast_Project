# Data Cleaning Documentation

เอกสารนี้อธิบายกระบวนการทำความสะอาดข้อมูล (Data Cleaning) ที่ใช้ใน Pipeline

---

## สรุปขั้นตอนการ Cleaning

ปัจจุบัน Pipeline มีการ Clean ข้อมูล **1 ขั้นตอน**:

1. ✅ **Handle Missing Values** - จัดการค่าที่หายไป (NaN)
2. ~~**Clip Outliers**~~ - ตัดค่าผิดปกติ (ถูกปิดใช้งานแล้ว)

---

## 1. Handle Missing Values (จัดการค่าที่หายไป)

### วิธีการทำงาน

ใช้เทคนิค **Forward Fill (ffill)** และ **Backward Fill (bfill)** เพื่อเติมค่าที่หายไป:

1. **Forward Fill**: เติมค่าที่หายด้วยค่าล่าสุดก่อนหน้า
2. **Backward Fill**: ถ้าข้อมูลช่วงต้นไม่มี ให้เติมด้วยค่าถัดไป
3. **Drop Rows**: ลบแถวที่ **ทุกคอลัมน์** เป็น NaN (ไม่ลบถ้ามีข้อมูลบางตัว)

### ตัวอย่างก่อน-หลัง

#### ก่อน Cleaning
```
         date       gold    oil  usd_thb
0  2024-01-01  2050.00    NaN   34.50
1  2024-01-02      NaN  75.20   34.52
2  2024-01-03  2055.00  75.50     NaN
3  2024-01-04      NaN    NaN     NaN   <- ทุกคอลัมน์เป็น NaN
4  2024-01-05  2060.00  76.00   34.60
```

#### หลัง Cleaning
```
         date       gold    oil  usd_thb
0  2024-01-01  2050.00  75.20   34.50   <- oil ถูก bfill จาก row 1
1  2024-01-02  2050.00  75.20   34.52   <- gold ถูก ffill จาก row 0
2  2024-01-03  2055.00  75.50   34.52   <- usd_thb ถูก ffill
3  (ถูกลบ)                              <- แถวนี้ถูกลบเพราะทุกค่าเป็น NaN
4  2024-01-05  2060.00  76.00   34.60
```

### เหตุผล
- ข้อมูลทางการเงินมักมีความต่อเนื่อง (ราคาเปลี่ยนแปลงช้า)
- Forward/Backward Fill เหมาะกับข้อมูลที่มี missing แบบสุ่ม
- เก็บแถวที่มีข้อมูลบางส่วนไว้ (ไม่ strict drop)

### Code
```python
from features.cleaning import handle_missing_values

cleaned_df = handle_missing_values(df, limit=None)
```

**Parameters:**
- `df`: DataFrame ที่ต้องการ clean
- `limit`: จำกัดจำนวนครั้งที่ fill (None = ไม่จำกัด)

---

## 2. Clip Outliers (ตัดค่าผิดปกติ) - ⚠️ ปิดใช้งาน

### สถานะ: **ไม่ใช้งานแล้ว**

Function นี้เคยใช้ในอดีตแต่ถูกปิดใช้งานแล้วเนื่องจาก:

### ปัญหาที่เกิด
1. **Gold Price Clipping**: ราคาทองคำปี 2025 พุ่งจาก ~2700 ไป ~4000+ แต่ถูก clip ลงเหลือ ~2734
2. **Trending Data**: ข้อมูลราคาหุ้น/สินค้าโภคภัณฑ์มักมี trend ที่เปลี่ยนแปลง การใช้ global outlier detection ไม่เหมาะสม

### วิธีการทำงาน (อดีต)
ใช้ **Winsorization** กับ **MAD-based Z-score**:

1. คำนวณ Median และ MAD (Median Absolute Deviation)
2. หา Z-score ของแต่ละค่า: `z = 0.6745 × (x - median) / MAD`
3. ถ้า |z| > 4.0 → ถือว่าเป็น outlier
4. Clip ค่า outlier ให้เท่ากับ 1st percentile (ขอบล่าง) หรือ 99th percentile (ขอบบน)

### ตัวอย่างปัญหาที่เกิด

**Gold Price 2020-2025**
```
ก่อน Clip (ข้อมูลจริง):
2020-2023: ~1800-2700
2025:      ~3500-4300 (ราคาพุ่งขึ้นจริงๆ)

หลัง Clip (ผิดพลาด):
2020-2023: ~1800-2700
2025:      ~2734 (ถูก clip เพราะถือว่าเป็น outlier!)
```

### เหตุผลที่ปิดใช้
- Financial time series ไม่ stationary (มี trend ที่เปลี่ยนไป)
- ราคาที่พุ่งขึ้น/ลงมากอาจเป็นการเคลื่อนไหวจริง ไม่ใช่ outlier
- Global outlier detection ไม่เหมาะกับข้อมูลที่มี structural breaks

### Code (ถ้าต้องการใช้เอง)
```python
from features.cleaning import clip_outliers

# ไม่แนะนำให้ใช้กับ price data
# clipped_df = clip_outliers(df, z_threshold=4.0)
```

---

## การเปลี่ยนแปลงใน Pipeline

### Timeline

**ก่อนวันที่ 2025-11-21:**
```python
# pipelines/run_pipeline.py
combined = handle_missing_values(combined)
combined = clip_outliers(combined)  # <- ทำให้ Gold price ผิดพลาด
```

**หลังวันที่ 2025-11-21:**
```python
# pipelines/run_pipeline.py
combined = handle_missing_values(combined)
# ลบ clip_outliers ออกแล้ว
```

### ผลลัพธ์
- ✅ USD/THB data ถูกต้อง (เปลี่ยนจาก Polygon → yfinance)
- ✅ Gold price ถูกต้อง (ไม่ถูก clip แล้ว)
- ✅ ข้อมูล time series สะท้อนความเป็นจริง

---

## ข้อควรระวัง

### 1. Missing Values
- ถ้าข้อมูลขาดหายมาก (>30%) อาจต้องพิจารณาวิธีอื่น
- Forward/Backward Fill อาจไม่เหมาะถ้ามี gap ยาวมาก

### 2. Outliers
- **ไม่ควร**ตัด outlier แบบ global กับ financial data
- ควรใช้ domain knowledge ในการตัดสินใจว่าค่าไหนผิดปกติจริงๆ
- ถ้าต้องการตรวจจับ anomaly ควรใช้ time series specific methods

---

## สรุป

| ขั้นตอน | สถานะ | วัตถุประสงค์ | ข้อควรระวัง |
|---------|-------|-------------|-------------|
| **Handle Missing Values** | ✅ ใช้งาน | เติมค่าที่หาย | ระวัง gap ยาว |
| **Clip Outliers** | ❌ ปิดใช้ | ตัดค่าผิดปกติ | ไม่เหมาะกับ trending data |

**ปัจจุบัน**: Pipeline ทำ cleaning เพียง **Missing Value Handling** เท่านั้น เพื่อรักษาความถูกต้องของข้อมูล time series
