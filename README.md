# DT514 — Data Classification (Week 5)

รวม Notebook, Dataset และ Model สำหรับการเรียนเรื่อง **Classification** ในรายวิชา DT514
เนื้อหาครอบคลุมตั้งแต่การเตรียมข้อมูล การแบ่ง Train/Test, การหา Hyper-parameter ด้วย Cross Validation,
การสร้างและเปรียบเทียบ Model หลายแบบด้วย Scikit-learn, การประเมินผลด้วย Metrics ต่าง ๆ
ไปจนถึงการบันทึก Model และเขียน Module สำหรับนำไปใช้พยากรณ์จริง

---

## สารบัญ

- [โครงสร้างโปรเจกต์](#โครงสร้างโปรเจกต์)
- [การติดตั้งและเริ่มใช้งาน](#การติดตั้งและเริ่มใช้งาน)
- [Dataset](#dataset)
- [รายละเอียดแต่ละ Notebook](#รายละเอียดแต่ละ-notebook)
  - [W5_classification.ipynb — WDBC (Breast Cancer)](#1-w5_classificationipynb--wdbc-breast-cancer)
  - [notebook_01.ipynb — Iris](#2-notebook_01ipynb--iris)
  - [notebook_02.ipynb — Titanic (ตัวอย่างจากชั้นเรียน)](#3-notebook_02ipynb--titanic-ตัวอย่างจากชั้นเรียน)
  - [notebook_titanic.ipynb — Titanic (ฉบับปรับปรุง)](#4-notebook_titanicipynb--titanic-ฉบับปรับปรุง)
- [สรุปผลการทดลอง](#สรุปผลการทดลอง)
- [การใช้งาน Model ที่บันทึกไว้ (model_02.mod)](#การใช้งาน-model-ที่บันทึกไว้-model_02mod)
- [แนวคิดสำคัญ](#แนวคิดสำคัญ)
- [ข้อควรระวังและปัญหาที่พบบ่อย](#ข้อควรระวังและปัญหาที่พบบ่อย)
- [License](#license)

---

## โครงสร้างโปรเจกต์

```
W5/
├── W5_classification.ipynb   # งานหลัก: เปรียบเทียบ 5 Models บน WDBC + Decision Tree เชิงลึก
├── notebook_01.ipynb         # ตัวอย่างที่ 1: Decision Tree บน Iris Dataset
├── notebook_02.ipynb         # ตัวอย่างที่ 2: Decision Tree บน Titanic Dataset (ต้นฉบับ)
├── notebook_titanic.ipynb    # Titanic ฉบับปรับปรุง: มี EDA, เทียบ Gini/Entropy, Prediction Module ครบ
├── iris.csv                  # Iris Dataset (150 แถว, 3 คลาส)
├── titanic.csv               # Titanic Dataset (1,043 แถว, 2 คลาส)
├── wdbc.csv                  # Wisconsin Diagnostic Breast Cancer (569 แถว, 2 คลาส)
├── wdbc.names                # คำอธิบาย Attribute ของ WDBC จาก UCI
├── model_02.mod              # Decision Tree (Titanic) ที่ train แล้ว บันทึกด้วย pickle
├── chapter_04.pdf            # เอกสารประกอบการเรียน บทที่ 4
├── LICENSE                   # MIT License
└── README.md
```

---

## การติดตั้งและเริ่มใช้งาน

### ความต้องการของระบบ

| Package | เวอร์ชันที่ใช้ทดสอบ |
|---|---|
| Python | 3.12.2 |
| numpy | 2.5.3 |
| pandas | 3.0.6 |
| scikit-learn | 1.9.1 |
| matplotlib | 3.11.2 |
| seaborn | 0.13.2 |
| ipykernel | 7.3.0 |

### ขั้นตอนการติดตั้ง

```bash
# 1. Clone repository
git clone <repository-url>
cd W5

# 2. สร้าง Virtual Environment (ไฟล์ .venv* ถูก ignore ไว้ใน .gitignore แล้ว)
python3 -m venv .venv-1
source .venv-1/bin/activate          # Windows: .venv-1\Scripts\activate

# 3. ติดตั้ง Package
pip install numpy pandas scikit-learn matplotlib seaborn ipykernel jupyter

# 4. เปิด Jupyter
jupyter notebook                      # หรือเปิดไฟล์ .ipynb ผ่าน VS Code แล้วเลือก Kernel เป็น .venv-1
```

> **หมายเหตุ:** `W5_classification.ipynb` และ `notebook_titanic.ipynb` อ่านไฟล์ CSV ด้วย **Absolute Path**
> (`/Users/pasunimsuwan/Desktop/DPU/DT-514/W5/...`) หากรันบนเครื่องอื่นให้แก้เป็น Relative Path เช่น
> `pd.read_csv('wdbc.csv', ...)` และ `pd.read_csv('titanic.csv')` ก่อนรัน

### ลำดับการรันที่แนะนำ

1. `notebook_01.ipynb` — เริ่มจากตัวอย่างง่ายที่สุด (Iris, 4 features)
2. `notebook_02.ipynb` / `notebook_titanic.ipynb` — เรียนรู้ Preprocessing, Cross Validation, การบันทึกและใช้งาน Model
3. `W5_classification.ipynb` — งานหลัก เปรียบเทียบหลาย Algorithm และวิเคราะห์ผลเชิงลึก

---

## Dataset

### 1. Iris (`iris.csv`)

| รายการ | รายละเอียด |
|---|---|
| จำนวนข้อมูล | 150 แถว |
| Features | `sepal_length`, `sepal_width`, `petal_length`, `petal_width` (ตัวเลขทั้งหมด หน่วย cm) |
| Target | `iris_class` — `Iris-setosa` (50), `Iris-versicolor` (50), `Iris-virginica` (50) |
| ประเภทปัญหา | Multi-class Classification (3 คลาส, สมดุล) |

### 2. Titanic (`titanic.csv`)

| คอลัมน์ | ความหมาย | ชนิด |
|---|---|---|
| `survived` | **Target** — 0 = เสียชีวิต (618), 1 = รอดชีวิต (425) | Category |
| `pclass` | ชั้นโดยสาร 1 / 2 / 3 | Category |
| `sex` | เพศ `male` / `female` | Category |
| `age` | อายุ (ปี) | Float |
| `sibsp` | จำนวนพี่น้อง/คู่สมรสที่เดินทางด้วย | Int |
| `parch` | จำนวนพ่อแม่/ลูกที่เดินทางด้วย | Int |
| `fare` | ค่าโดยสาร | Float |
| `embarked` | ท่าเรือที่ขึ้น `C` = Cherbourg, `Q` = Queenstown, `S` = Southampton | Category |

- จำนวนข้อมูล 1,043 แถว ไม่มี Missing Value
- ไฟล์มี BOM (`﻿`) ที่หัวคอลัมน์แรก — pandas อ่านได้ปกติ

### 3. Wisconsin Diagnostic Breast Cancer — WDBC (`wdbc.csv`, `wdbc.names`)

| รายการ | รายละเอียด |
|---|---|
| จำนวนข้อมูล | 569 แถว |
| Target | `diagnosis` — `B` = Benign / ไม่ใช่มะเร็ง (357), `M` = Malignant / มะเร็ง (212) |
| Features | 30 ค่า คำนวณจากภาพถ่ายดิจิทัลของ Fine-Needle Aspirate (FNA) ของก้อนเนื้อ |
| ที่มา | UCI Machine Learning Repository (รายละเอียดใน `wdbc.names`) |

Features 30 ค่ามาจากลักษณะของนิวเคลียสเซลล์ 10 ด้าน แต่ละด้านคำนวณ 3 แบบ:

| ลักษณะ (10 ด้าน) | `m_` = Mean | `se_` = Standard Error | `w_` = Worst (ค่าเฉลี่ย 3 ค่าที่มากที่สุด) |
|---|---|---|---|
| radius, texture, perimeter, area, smoothness, compactness, concavity, concave_points, symmetry, fractal_dimension | ✓ | ✓ | ✓ |

ใน Notebook จะเปลี่ยนชื่อคอลัมน์ให้อ่านง่าย เช่น `m_radius` → `Mean Radius`, `se_area` → `Area Error`, `w_concave_points` → `Worst Concave Points`

---

## รายละเอียดแต่ละ Notebook

### 1. `W5_classification.ipynb` — WDBC (Breast Cancer)

งานหลักของสัปดาห์ ตอบ **5 คำถามหลัก** ของ Classification:

1. Classification Problem คืออะไร?
2. ทำไม Machine Learning ถึงแก้ปัญหา Classification ได้?
3. ใช้เทคนิคและเครื่องมืออะไร?
4. สร้าง Classification Model ได้อย่างไร?
5. Model ดีพอหรือไม่?

| Section | เนื้อหา |
|---|---|
| **1. โหลดและตรวจสอบ Dataset** | โหลด `wdbc.csv` พร้อมตั้งชื่อคอลัมน์ใหม่, ดู `head()`, shape, `describe()`, ตรวจ Missing Value · แสดงกราฟ Pie (สัดส่วน M/B), Histogram ของ Mean Radius และ Box Plot ของ Mean Area แยกตาม Diagnosis |
| **2. Preprocess Data** | แยก `X` (30 features) และ `y` · `LabelEncoder` แปลง B→0, M→1 · `StandardScaler` ปรับให้ mean = 0, std = 1 · Correlation Heatmap ของทุก feature |
| **3. Split Dataset** | `train_test_split` 80/20, `random_state=42`, `stratify=y` → Train 455 (B 285 / M 170), Test 114 (B 72 / M 42) |
| **4. Train Multiple Models** | Decision Tree (`max_depth=10`), KNN (`k=5`), Logistic Regression (`max_iter=1000`), Random Forest (`n_estimators=100`), SVM (RBF kernel) |
| **5. Evaluate Models** | คำนวณ Accuracy / Precision / Recall / F1 ของทุก Model · Bar Chart เปรียบเทียบ · Confusion Matrix ของ Model ที่ดีที่สุด, Decision Tree และ KNN · Classification Report · Feature Importance จาก Random Forest (Top 15) |
| **6. Decision Tree แบบละเอียด** | ใช้ข้อมูลดิบ (ไม่ Scale) เพื่อให้อ่าน threshold เป็นหน่วยจริง · หา `max_depth` ที่ดีที่สุด (1–15) จากกราฟ Train vs Test Accuracy · Train ใหม่ด้วย depth ที่ดีที่สุด · `plot_tree` (แสดง 3 ชั้นแรก) · `export_text` แสดงกฎการตัดสินใจ · Feature Importance ของ Decision Tree |
| **สรุปผล** | ตอบ 5 คำถามหลักพร้อมตัวเลขจากการทดลอง |

### 2. `notebook_01.ipynb` — Iris

ตัวอย่างพื้นฐานของ Decision Tree แบบ Multi-class

| Step | เนื้อหา |
|---|---|
| 0. Setting Environment | Import numpy, pandas, matplotlib และ sklearn |
| 1. Prepare Dataset | โหลด `iris.csv` และ map ชื่อคลาสเป็นตัวเลข (`setosa`=0, `versicolor`=1, `virginica`=2) |
| 2. Split Data | `train_test_split(test_size=0.33, random_state=42)` |
| 3. Find Best Hyper-parameters | วน `max_depth` 1–9 ด้วย 5-Fold `cross_validate` วัด Accuracy / Precision / Recall / F1 แบบ macro แล้ว plot กราฟ |
| 4. Create Model | `DecisionTreeClassifier(max_depth=3)` |
| 5. Test Model | Confusion Matrix + Metrics รายคลาส |
| 6. Visualize Model | `tree.plot_tree` แสดงโครงสร้างต้นไม้ |

### 3. `notebook_02.ipynb` — Titanic (ตัวอย่างจากชั้นเรียน)

ตัวอย่าง Workflow ครบวงจรตั้งแต่ข้อมูลดิบจนถึง Module สำหรับ Production

| Section | เนื้อหา |
|---|---|
| 01 Import Libraries | รวม `pickle` สำหรับบันทึก Model |
| 02 Preprocess Data | กำหนด dtype · One-hot Encoding (`pclass`, `sex`, `embarked`) · Min-Max Normalization (`age`, `sibsp`, `parch`, `fare`) · Split 67/33 (`random_state=45`) |
| 03 Cross Validation | ทดสอบ Entropy `max_depth=3` และวน `max_depth` 1–30 ด้วย Gini |
| 04 Build Model | `DecisionTreeClassifier(criterion='entropy', max_depth=4)` · Train บนข้อมูลทั้งหมดแล้วบันทึกเป็น `model_02.mod` |
| 05 Extract Knowledge | `plot_tree`, `export_text` และฟังก์ชัน `get_rules()` แปลงต้นไม้เป็นกฎ if-then เรียงตามจำนวนตัวอย่าง |
| 06 Preprocessing Module | ฟังก์ชัน `feature_engineer()` แปลงข้อมูลดิบ 1 แถวให้อยู่ในรูปแบบเดียวกับตอน Train |
| 07 Prediction Module | โหลด `model_02.mod` แล้วพยากรณ์ |

### 4. `notebook_titanic.ipynb` — Titanic (ฉบับปรับปรุง)

เนื้อหาเดียวกับ `notebook_02.ipynb` แต่ปรับปรุงให้ครบและอ่านง่ายขึ้น:

- เพิ่ม **Section 02: Load and Explore Data** (`info()`, ตรวจ Missing Value)
- เปรียบเทียบ **Gini vs Entropy** โดย plot กราฟ `max_depth` 1–30 แยกกันทั้งสองแบบ
- ตั้งชื่อคลาสใน Report เป็น `Died` / `Survived`
- ฟังก์ชัน `get_rules()` ใช้ชื่อ feature จริงและแสดงผลเป็นรายการลำดับเลข
- **Section 08: Prediction Module** — `model_predict(X_raw)` รับ DataFrame ดิบ → เรียก `feature_engineer()` ทีละแถว → พยากรณ์ด้วย Model ที่โหลดจาก `model_02.mod`

---

## สรุปผลการทดลอง

### WDBC — เปรียบเทียบ 5 Models (Test set 114 ตัวอย่าง, Positive = Malignant)

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Decision Tree (`max_depth=10`) | 0.9298 | 0.9048 | 0.9048 | 0.9048 |
| KNN (`k=5`) | 0.9561 | 0.9744 | 0.9048 | 0.9383 |
| Logistic Regression | 0.9649 | 0.9750 | 0.9286 | 0.9512 |
| **Random Forest** | **0.9737** | **1.0000** | **0.9286** | **0.9630** |
| **SVM (RBF)** | **0.9737** | **1.0000** | **0.9286** | **0.9630** |

- Random Forest และ SVM ได้ผลเท่ากัน — Notebook เลือก **Random Forest** เป็น Best Model (ตัวแรกที่ได้ Accuracy สูงสุด)
- Random Forest พยากรณ์ Malignant ไม่ผิดเลย (Precision 100%) แต่พลาด Malignant ไป 3 จาก 42 ราย (Recall 92.86%)
- **Top 5 Features (Random Forest):** Worst Area (0.151), Worst Concave Points (0.127), Worst Radius (0.094), Worst Perimeter (0.084), Mean Concave Points (0.081)

### WDBC — Decision Tree เชิงลึก (Section 6)

- `max_depth` ที่ดีที่สุด = **7** (Test Accuracy 0.9386) — ต้นไม้มี 22 Leaf
- เมื่อ depth เพิ่มขึ้น Train Accuracy เข้าใกล้ 1.0 แต่ Test Accuracy ไม่เพิ่มตาม → เกิด **Overfitting**
- Decision Tree ใช้เพียง **14 จาก 30 Features** โดย `Worst Perimeter` มีความสำคัญสูงถึง 0.737 และเป็นเงื่อนไขแรกที่ราก (`Worst Perimeter <= 112.80`)

### Iris — Decision Tree (`max_depth=3`)

| Metric | ค่า |
|---|---|
| Accuracy | 0.98 |
| Precision (setosa / versicolor / virginica) | 1.00 / 0.94 / 1.00 |
| Recall (setosa / versicolor / virginica) | 1.00 / 1.00 / 0.94 |

### Titanic — Decision Tree (`criterion='entropy'`, `max_depth=4`)

| | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Died (0) | 0.79 | 0.87 | 0.83 | 191 |
| Survived (1) | 0.82 | 0.71 | 0.76 | 154 |
| **Accuracy** | | | **0.80** | 345 |

5-Fold CV บน Training set (Entropy, `max_depth=3`): Accuracy 0.798, Precision 0.781, Recall 0.668, F1-macro 0.779

---

## การใช้งาน Model ที่บันทึกไว้ (`model_02.mod`)

`model_02.mod` คือ `DecisionTreeClassifier(criterion='entropy', max_depth=4)` ที่ Train ด้วยข้อมูล Titanic ทั้งหมด (บันทึกด้วย `pickle`)
Input ต้องผ่าน `feature_engineer()` ก่อน เพื่อให้ได้ 12 features ตามลำดับนี้:

```
age, sibsp, parch, fare,
pclass_1, pclass_2, pclass_3,
sex_female, sex_male,
embarked_C, embarked_Q, embarked_S
```

การ Normalize ใน `feature_engineer()` ใช้ค่า min/max ที่ fix ไว้จากข้อมูล Train:

| Feature | สูตร |
|---|---|
| age | `(age - 0.1667) / (80.0 - 0.1667)` |
| sibsp | `sibsp / 8` |
| parch | `parch / 6` |
| fare | `fare / 512.3292` |

ตัวอย่างการใช้งาน (หลังจากรัน cell ที่นิยาม `feature_engineer()` ใน `notebook_titanic.ipynb` แล้ว):

```python
import pickle
import pandas as pd

model = pickle.load(open('model_02.mod', 'rb'))

passenger = pd.Series({
    'pclass': 1, 'sex': 'female', 'age': 29.0,
    'sibsp': 0, 'parch': 0, 'fare': 211.34, 'embarked': 'S',
})
x = pd.DataFrame([feature_engineer(passenger)])
print(model.predict(x))        # 0 = Died, 1 = Survived
```

> ⚠️ ไฟล์ pickle ผูกกับเวอร์ชันของ scikit-learn ที่ใช้บันทึก (1.9.1) หากโหลดด้วยเวอร์ชันอื่นอาจมี Warning หรือ Error — ให้รัน Notebook เพื่อ Train และบันทึกใหม่
> และควรโหลดไฟล์ pickle จากแหล่งที่เชื่อถือได้เท่านั้น

---

## แนวคิดสำคัญ

### Workflow ของ Classification

```
โหลดข้อมูล → สำรวจข้อมูล (EDA) → Preprocess → Split Train/Test
    → หา Hyper-parameter (Cross Validation) → Train → Evaluate → บันทึก Model → ใช้งานจริง
```

### Preprocessing

| เทคนิค | ใช้เมื่อ | ใช้ใน |
|---|---|---|
| Label Encoding | แปลง Target ที่เป็นข้อความเป็นตัวเลข | Iris, WDBC |
| One-hot Encoding | Feature แบบ Category ที่ไม่มีลำดับ | Titanic |
| Min-Max Normalization | ปรับค่าให้อยู่ในช่วง 0–1 | Titanic |
| Standardization (Z-score) | ปรับให้ mean = 0, std = 1 — จำเป็นสำหรับ KNN, SVM, Logistic Regression | WDBC |

> Decision Tree และ Random Forest **ไม่จำเป็นต้อง Scale** เพราะแบ่งข้อมูลด้วย threshold ทีละ feature

### Metrics

| Metric | สูตร | ความหมาย |
|---|---|---|
| Accuracy | (TP + TN) / (TP + TN + FP + FN) | สัดส่วนที่พยากรณ์ถูกทั้งหมด |
| Precision | TP / (TP + FP) | จากที่พยากรณ์ว่าเป็น Positive ถูกจริงกี่ % (FP น้อย = ไม่เตือนผิด) |
| Recall | TP / (TP + FN) | จาก Positive จริงทั้งหมด จับได้กี่ % (FN น้อย = ไม่พลาด) |
| F1-Score | 2 × (Precision × Recall) / (Precision + Recall) | ค่าเฉลี่ยฮาร์มอนิกที่สมดุลระหว่าง Precision และ Recall |

> ในงานทางการแพทย์อย่าง WDBC **Recall สำคัญที่สุด** เพราะการพลาดผู้ป่วยมะเร็ง (False Negative) มีผลร้ายแรงกว่าการเตือนผิด

### Decision Tree Hyper-parameters

- **`criterion`** — `gini` (Gini Impurity) หรือ `entropy` (Information Gain) ใช้วัดความ "ไม่บริสุทธิ์" ของโหนด (0 = มีคลาสเดียว)
- **`max_depth`** — ความลึกสูงสุดของต้นไม้ ลึกเกินไป → Overfitting, ตื้นเกินไป → Underfitting

---

## ข้อควรระวังและปัญหาที่พบบ่อย

- **Absolute Path** — `W5_classification.ipynb` และ `notebook_titanic.ipynb` ใช้ path เต็มของเครื่องผู้เขียน ต้องแก้ก่อนรันบนเครื่องอื่น
- **Data Leakage** — ใน Notebook มีการ `fit` Scaler กับข้อมูลทั้งหมดก่อน Split ซึ่งทำเพื่อความง่ายในการสอน ในงานจริงควร `fit` กับ Training set เท่านั้นแล้วค่อย `transform` Test set (หรือใช้ `Pipeline`)
- **การเลือก `max_depth` จาก Test set** — Section 6 ของ WDBC เลือก depth จาก Test Accuracy เพื่อการสาธิต ในทางปฏิบัติควรเลือกด้วย Cross Validation บน Training set แบบใน `notebook_01` / `notebook_02`
- **`notebook_02.ipynb` Section 07** — ฟังก์ชัน `model_predic()` ใช้ตัวแปร global `df_final` แทน argument `X` ที่รับเข้ามา ดูเวอร์ชันที่ถูกต้องได้ใน `notebook_titanic.ipynb` (`model_predict()`)
- **Warning ที่อาจพบ**
  - `SVC(probability=True)` ถูก deprecate ใน scikit-learn 1.9 — แนะนำให้ใช้ `CalibratedClassifierCV(SVC(), ensemble=False)` แทน
  - seaborn เตือนเรื่องการส่ง `palette` โดยไม่กำหนด `hue` — แก้ได้โดยเพิ่ม `hue='Diagnosis', legend=False`
  - Font Arial ไม่มีตัวอักษรไทย ทำให้ข้อความไทยในกราฟแสดงเป็นกล่องสี่เหลี่ยม — เปลี่ยนเป็น Font ที่รองรับภาษาไทย เช่น `Tahoma` หรือ `Thonburi` (macOS)

---

## License

เผยแพร่ภายใต้ [MIT License](LICENSE) — Copyright (c) 2026 Pasu Nimsuwan
