# 🛡️ ShelfGuard: The Financial Operating System for CPG Brands

> **Stop optimizing ROAS. Start optimizing Capital Efficiency.**

ShelfGuard is an autonomous capital allocation engine for modern consumer brands. It triangulates **Advertising Data** (Amazon Ads), **Inventory Logistics** (SP-API), and **Competitive Intelligence** (Keepa) to act as an automated CFO—directing every dollar of ad spend to where it generates the highest Net Margin, not just the most clicks.

---

## 📉 The Problem: The "Silo Trap"
Current e-commerce tools are fragmented, leading to massive capital inefficiency:
* **Ad Managers (Pacvue, Perpetua):** Optimize for *Spend* (ROAS), ignoring inventory health.
* **Ops Tools (SoStocked):** Optimize for *Stock*, ignoring marketing pushes.
* **Finance Tools (A2X):** Look *backward* at last month's P&L.

**Result:** Brands spend thousands on ads for products that are out of stock, low margin, or losing the Buy Box.

## 🚀 The Solution: ShelfGuard
ShelfGuard sits above these silos. It ingests data from all three sources to answer the single most important question: **"Where should my next dollar go?"**

### Core Capabilities

#### 1. 🏦 Strategic Capital Zones
Instead of managing "Campaigns," ShelfGuard classifies your entire catalog into three investment zones:
* **🏰 FORTRESS (Maintenance):** High Share, Healthy Stock. *Action: Protect Margins.*
* **🚀 FRONTIER (Growth):** Good Fundamentals, Low Share. *Action: Aggressive Investment.*
* **📉 DRAG (Waste):** Low Share, Low Inventory. *Action: Immediate Divestment.*

#### 2. 🚦 The Omni-Channel Traffic Router
ShelfGuard monitors "Ground Truth" (Buy Box Status & Inventory Levels) to direct external traffic:
* **Scenario A:** Buy Box is Won & In Stock → **Drive Traffic to Amazon** (Flywheel Effect).
* **Scenario B:** Buy Box Lost (Price War) → **Divert Traffic to Shopify/DTC** (Protect Margin).

#### 3. 📉 The "Efficiency Score" Algorithm
A proprietary 0-100 score that quantifies portfolio health.
* *Inputs:* `BuyBox%`, `WeeksOfCover`, `MarginGap`.
* *Output:* A single metric for the CFO/Board to track capital efficiency risk.

---

## 🛠️ Tech Stack
Built for speed, modularity, and data integrity.

* **Core Logic:** Python 3.10+ (Pandas, NumPy)
* **Frontend:** Streamlit (Real-time Interactive Dashboard)
* **Backend / Database:** Supabase (PostgreSQL)
* **Data Pipelines:**
    * `Amazon SP-API` (Sales & Inventory)
    * `Amazon Advertising API` (Spend & Performance)
    * `Keepa API` (Competitor Intelligence)
* **Authentication:** Local Environment (`.env` protected)

---

## ⚡ Quick Start (Local Dev)

**1. Clone the Repository**
```bash
git clone [https://github.com/jshuck0/ShelfGuard.git](https://github.com/jshuck0/ShelfGuard.git)
cd ShelfGuard
```

**2. Set up Environment**
Create a virtual environment to keep dependencies clean:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

**3. Install Dependencies**
```bash
pip install -r requirements.txt
```

**4. Configure Secrets**
Create a `.env` file in the root directory (do NOT commit this):
```ini
SUPABASE_URL=your_url_here
SUPABASE_KEY=your_key_here
KEEPA_API_KEY=your_key_here
```

**5. Run the Financial OS**
```bash
streamlit run apps/shelfguard_app.py
```

---

## 📄 License
Private Proprietary Software. All rights reserved.