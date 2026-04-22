from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
from datetime import timedelta
from typing import Optional, Dict, Any
import os
import json
from dotenv import load_dotenv
load_dotenv()

from langchain.agents import create_agent
from langchain_nvidia_ai_endpoints import ChatNVIDIA


# -----------------------------
# APP INIT
# -----------------------------
app = FastAPI(title="Retail ML Platform")
templates = Jinja2Templates(directory="templates")


# -----------------------------
# LOAD DATA
# -----------------------------
forecast_df = pd.read_csv("forecast_dataset.csv")
store_kpi_df = pd.read_csv("store_kpis.csv")
customer_kpi_df = pd.read_csv("c2.csv")
customer_df = pd.read_csv("cc.csv", dtype={"StateHoliday": "str"})

forecast_df["Date"] = pd.to_datetime(forecast_df["Date"], errors="coerce")
forecast_df = forecast_df.dropna(subset=["Date"]).copy()

for df in [forecast_df, store_kpi_df, customer_kpi_df, customer_df]:
    if "Store" in df.columns:
        df["Store"] = pd.to_numeric(df["Store"], errors="coerce")

if "Date" in customer_df.columns:
    customer_df["Date"] = pd.to_datetime(customer_df["Date"], errors="coerce")

if "First_Purchase_Date" in customer_df.columns:
    customer_df["First_Purchase_Date"] = pd.to_datetime(
        customer_df["First_Purchase_Date"], errors="coerce"
    )

if "Last_Purchase_Date" in customer_df.columns:
    customer_df["Last_Purchase_Date"] = pd.to_datetime(
        customer_df["Last_Purchase_Date"], errors="coerce"
    )


# -----------------------------
# LOAD MODELS
# -----------------------------
forecast_model = joblib.load("sales_forecast_model.pkl")
forecast_features = joblib.load("forecast_features.pkl")

# Prefer unified churn pipeline, fallback to separate model/features
if os.path.exists("churn_pipeline.pkl"):
    churn_pipeline = joblib.load("churn_pipeline.pkl")
    churn_model = churn_pipeline["model"]
    churn_features = churn_pipeline["features"]
else:
    churn_model = joblib.load("churn_model.pkl")
    churn_features = joblib.load("churn_features.pkl")


# -----------------------------
# CHAT STATE
# -----------------------------
latest_upload_analysis: Dict[str, Any] = {}

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
chat_agent = None
llm = None

if NVIDIA_API_KEY:
    llm = ChatNVIDIA(
        model="meta/llama-3.1-70b-instruct",
        api_key=NVIDIA_API_KEY,
        temperature=0
    )


# -----------------------------
# HELPERS
# -----------------------------
ASSORTMENT_DECODE = {0: "a", 1: "b", 2: "c"}
STORETYPE_DECODE = {0: "a", 1: "b", 2: "c", 3: "d"}


def to_python(value):
    if pd.isna(value):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def decode_category(value, mapping):
    if pd.isna(value):
        return "N/A"

    # if already string like 'a', 'b', keep it
    if isinstance(value, str):
        return value

    try:
        return mapping.get(int(value), str(value))
    except Exception:
        return str(value)


def normalize_stateholiday(series: pd.Series) -> pd.Series:
    mapping = {
        "0": 0, 0: 0,
        "a": 1, "A": 1,
        "b": 2, "B": 2,
        "c": 3, "C": 3,
        "d": 4, "D": 4
    }
    return series.map(lambda x: mapping.get(x, x)).fillna(0).astype(int)


def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"]).copy()
        df["Year"] = df["Date"].dt.year.astype(int)
        df["Month"] = df["Date"].dt.month.astype(int)
        df["Week"] = df["Date"].dt.isocalendar().week.astype(int)
        df["DayOfWeek"] = df["Date"].dt.dayofweek.astype(int)
        df["IsWeekend"] = (df["DayOfWeek"] >= 5).astype(int)
    return df


def add_sales_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Sales" in df.columns:
        df["Sales"] = pd.to_numeric(df["Sales"], errors="coerce")
        df = df.dropna(subset=["Sales"]).copy()

        df["Sales_Lag_1"] = df["Sales"].shift(1).bfill().fillna(0)
        df["Sales_Lag_7"] = df["Sales"].shift(7).bfill().fillna(0)
        df["Sales_Lag_30"] = df["Sales"].shift(30).bfill().fillna(0)
        df["Rolling_Mean_7"] = df["Sales"].rolling(window=7, min_periods=1).mean()
        df["Rolling_Mean_30"] = df["Sales"].rolling(window=30, min_periods=1).mean()

    return df


def ensure_forecast_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = add_date_features(df)

    if "StateHoliday" in df.columns:
        df["StateHoliday"] = normalize_stateholiday(df["StateHoliday"])
    else:
        df["StateHoliday"] = 0

    if "Promo" not in df.columns:
        df["Promo"] = 0
    df["Promo"] = pd.to_numeric(df["Promo"], errors="coerce").fillna(0).astype(int)

    if "SchoolHoliday" not in df.columns:
        df["SchoolHoliday"] = 0
    df["SchoolHoliday"] = pd.to_numeric(df["SchoolHoliday"], errors="coerce").fillna(0).astype(int)

    if "Store" not in df.columns:
        df["Store"] = 1
    df["Store"] = pd.to_numeric(df["Store"], errors="coerce").fillna(1).astype(int)

    df = add_sales_features(df)

    for col in forecast_features:
        if col not in df.columns:
            df[col] = 0

    return df


def forecast_next_days(model, data: pd.DataFrame, days: int = 7):
    predictions = []

    df = data.copy().sort_values("Date").reset_index(drop=True)
    df = ensure_forecast_columns(df)

    if df.empty:
        return predictions

    sales_history = df["Sales"].tolist()
    last_row = df.tail(1).copy()

    for _ in range(days):
        for col in forecast_features:
            if col not in last_row.columns:
                last_row[col] = 0

        X = last_row[forecast_features]
        pred = float(model.predict(X)[0])
        pred = round(pred, 2)
        predictions.append(pred)

        sales_history.append(pred)

        next_row = last_row.copy()
        next_row["Date"] = next_row["Date"] + timedelta(days=1)

        next_row["Year"] = next_row["Date"].dt.year.astype(int)
        next_row["Month"] = next_row["Date"].dt.month.astype(int)
        next_row["Week"] = next_row["Date"].dt.isocalendar().week.astype(int)
        next_row["DayOfWeek"] = next_row["Date"].dt.dayofweek.astype(int)
        next_row["IsWeekend"] = (next_row["DayOfWeek"] >= 5).astype(int)

        next_row["Sales_Lag_1"] = sales_history[-1]
        next_row["Sales_Lag_7"] = sales_history[-7] if len(sales_history) >= 7 else sales_history[-1]
        next_row["Sales_Lag_30"] = sales_history[-30] if len(sales_history) >= 30 else sales_history[-1]
        next_row["Rolling_Mean_7"] = float(np.mean(sales_history[-7:]))
        next_row["Rolling_Mean_30"] = float(np.mean(sales_history[-30:]))

        next_row["Promo"] = int(next_row["Promo"].iloc[0]) if "Promo" in next_row.columns else 0
        next_row["SchoolHoliday"] = int(next_row["SchoolHoliday"].iloc[0]) if "SchoolHoliday" in next_row.columns else 0
        next_row["StateHoliday"] = int(next_row["StateHoliday"].iloc[0]) if "StateHoliday" in next_row.columns else 0

        last_row = next_row

    return predictions


def build_churn_features(payload: Dict[str, Any]) -> pd.DataFrame:
    data = dict(payload)

    if data.get("First_Purchase_Date"):
        dt = pd.to_datetime(data["First_Purchase_Date"], errors="coerce")
        if pd.notna(dt):
            data["First_Purchase_Date_year"] = int(dt.year)
            data["First_Purchase_Date_month"] = int(dt.month)
            data["First_Purchase_Date_day"] = int(dt.day)
            data["First_Purchase_Date_weekday"] = int(dt.weekday())

    row = {}
    for col in churn_features:
        row[col] = data.get(col, 0)

    input_df = pd.DataFrame([row])

    for col in input_df.columns:
        input_df[col] = pd.to_numeric(input_df[col], errors="coerce").fillna(0)

    return input_df


# -----------------------------
# REQUEST SCHEMAS
# -----------------------------
class ChurnPredictionInput(BaseModel):
    Total_Transactions: float = 0
    Total_Sales: float = 0
    Promo_Usage: float = 0
    Store_Visits: float = 0
    Days_Since_First_Purchase: float = 0
    Avg_Basket_Size: float = 0
    Customer_Tenure: float = 0
    Promo_Usage_Rate: float = 0
    Frequency: float = 0
    Monetary: float = 0
    Store: Optional[int] = 0
    StoreType: Optional[int] = 0
    Assortment: Optional[int] = 0

    First_Purchase_Date: Optional[str] = None
    First_Purchase_Date_year: Optional[int] = 0
    First_Purchase_Date_month: Optional[int] = 0
    First_Purchase_Date_day: Optional[int] = 0
    First_Purchase_Date_weekday: Optional[int] = 0


class ChatRequest(BaseModel):
    message: str
    store_id: Optional[int] = None
    page: Optional[str] = None


# -----------------------------
# PAGE ROUTES
# -----------------------------
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(request=request, name="home.html")


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard_page(request: Request):
    return templates.TemplateResponse(request=request, name="dashboard.html")


@app.get("/customers", response_class=HTMLResponse)
def customers_page(request: Request):
    return templates.TemplateResponse(request=request, name="customers.html")


@app.get("/upload", response_class=HTMLResponse)
def upload_page(request: Request):
    return templates.TemplateResponse(request=request, name="upload.html")


@app.get("/compare", response_class=HTMLResponse)
def compare_page(request: Request):
    return templates.TemplateResponse(request=request, name="compare.html")


@app.get("/explore", response_class=HTMLResponse)
def explore_page(request: Request):
    return templates.TemplateResponse(request=request, name="explore.html")


# -----------------------------
# STORE APIs
# -----------------------------
@app.get("/stores")
def get_stores():
    forecast_stores = set(
        forecast_df["Store"].dropna().astype(int).tolist()
    )

    customer_stores = set(
        customer_df["Store"].dropna().astype(int).tolist()
    )

    common_stores = sorted(forecast_stores.intersection(customer_stores))
    return common_stores


@app.get("/api/store-history/{store_id}")
def store_history(store_id: int):
    df = forecast_df[forecast_df["Store"] == store_id].sort_values("Date").copy()

    if df.empty:
        return {"error": "Store not found"}

    recent_df = df.tail(30)

    return {
        "dates": recent_df["Date"].dt.strftime("%Y-%m-%d").tolist(),
        "sales": [round(float(x), 2) for x in recent_df["Sales"].fillna(0).tolist()],
        "customers": [int(x) for x in recent_df["Customers"].fillna(0).tolist()] if "Customers" in recent_df.columns else []
    }


@app.get("/store-dashboard/{store_id}")
def store_dashboard(store_id: int):
    df = forecast_df[forecast_df["Store"] == store_id].copy()
    kpi_row = store_kpi_df[store_kpi_df["Store"] == store_id].copy()

    if df.empty or kpi_row.empty:
        return {"error": "Store not found"}

    kpi_row = kpi_row.iloc[0]

    best_assortment = (
        decode_category(kpi_row["Best_Assortment"], ASSORTMENT_DECODE)
        if "Best_Assortment" in kpi_row.index else "N/A"
    )

    best_store_type = (
        decode_category(kpi_row["Best_StoreType"], STORETYPE_DECODE)
        if "Best_StoreType" in kpi_row.index else "N/A"
    )

    kpis = {
        "Total_Sales": round(float(kpi_row["Total_Sales"]), 2) if "Total_Sales" in kpi_row.index else 0.0,
        "Avg_Daily_Sales": round(float(kpi_row["Avg_Daily_Sales"]), 2) if "Avg_Daily_Sales" in kpi_row.index else 0.0,
        "Store_Performance_Index": round(float(kpi_row["Store_Performance_Index"]), 2) if "Store_Performance_Index" in kpi_row.index else 0.0,
        "Best_Assortment": best_assortment,
        "Best_StoreType": best_store_type,
        "Total_Customers": int(kpi_row["Total_Customers"]) if "Total_Customers" in kpi_row.index and pd.notna(kpi_row["Total_Customers"]) else 0,
        "Promo_Usage_Rate": round(float(kpi_row["Promo_Usage_Rate"]), 2) if "Promo_Usage_Rate" in kpi_row.index else 0.0,
        "Total_Open_Days": int(kpi_row["Total_Open_Days"]) if "Total_Open_Days" in kpi_row.index and pd.notna(kpi_row["Total_Open_Days"]) else 0,
        "Avg_Customers": round(float(kpi_row["Avg_Customers"]), 2) if "Avg_Customers" in kpi_row.index and pd.notna(kpi_row["Avg_Customers"]) else 0.0,
        "Promo_Days": int(kpi_row["Promo_Days"]) if "Promo_Days" in kpi_row.index and pd.notna(kpi_row["Promo_Days"]) else 0,
        "Sales_per_Customer": round(float(kpi_row["Sales_per_Customer"]), 2) if "Sales_per_Customer" in kpi_row.index and pd.notna(kpi_row["Sales_per_Customer"]) else 0.0,
        "Sales_Rank": int(kpi_row["Sales_Rank"]) if "Sales_Rank" in kpi_row.index and pd.notna(kpi_row["Sales_Rank"]) else 0,
        "Customer_Productivity": round(float(kpi_row["Customer_Productivity"]), 2) if "Customer_Productivity" in kpi_row.index and pd.notna(kpi_row["Customer_Productivity"]) else 0.0,
    }

    forecast = forecast_next_days(forecast_model, df, 7)

    return {"kpis": kpis, "forecast": forecast}


# -----------------------------
# CUSTOMER / CHURN APIs
# -----------------------------
@app.get("/customer-kpis/{store_id}")
def customer_kpis(store_id: int):
    row = customer_kpi_df[customer_kpi_df["Store"] == store_id].copy()

    if row.empty:
        return {"error": "Customer KPI data not found for this store"}

    row = row.iloc[0]
    response = {}

    for col in row.index:
        if col == "Store":
            continue

        value = row[col]
        response[col] = to_python(value)

    return response


@app.get("/customer-kpis-by-store")
def customer_kpis_by_store():
    records = customer_kpi_df.copy()
    for col in records.columns:
        records[col] = records[col].apply(to_python)
    return records.to_dict(orient="records")


@app.get("/churn-kpis/{store_id}")
def churn_kpis(store_id: int):
    if "Store" not in customer_df.columns:
        return {"error": "Store column not found in customer_churn_dataset.csv"}

    if "Churn" not in customer_df.columns:
        return {"error": "Churn column not found in customer_churn_dataset.csv"}

    df = customer_df[customer_df["Store"] == store_id].copy()

    if df.empty:
        return {"error": f"No churn data found for store {store_id}"}

    total_customers = int(df.shape[0])
    churn_rate = round(float(df["Churn"].mean()), 3)
    active_customers = int((df["Churn"] == 0).sum())
    churned_customers = int((df["Churn"] == 1).sum())

    return {
        "Total_Customers": total_customers,
        "Churn_Rate": churn_rate,
        "Active_Customers": active_customers,
        "Churned_Customers": churned_customers
    }


@app.get("/promo-churn/{store_id}")
def promo_churn(store_id: int):
    if "Promo" not in customer_df.columns or "Churn" not in customer_df.columns:
        return {"error": "Promo or Churn column not found"}

    if "Store" not in customer_df.columns:
        return {"error": "Store column not found"}

    df = customer_df[customer_df["Store"] == store_id].copy()

    if df.empty:
        return {"error": f"No data for store {store_id}"}

    df["Promo"] = pd.to_numeric(df["Promo"], errors="coerce").fillna(0)
    df["Churn"] = pd.to_numeric(df["Churn"], errors="coerce").fillna(0)

    promo_users = df[df["Promo"] > 0]
    non_promo_users = df[df["Promo"] == 0]

    promo_churn_rate = float(promo_users["Churn"].mean()) if not promo_users.empty else 0.0
    non_promo_churn_rate = float(non_promo_users["Churn"].mean()) if not non_promo_users.empty else 0.0

    return {
        "Promo_Users_Churn_Rate": round(promo_churn_rate, 4),
        "Non_Promo_Users_Churn_Rate": round(non_promo_churn_rate, 4),
        "Promo_Users": int(len(promo_users)),
        "Non_Promo_Users": int(len(non_promo_users))
    }

# -----------------------------
# CHURN PREDICTION API
# -----------------------------
@app.post("/predict-churn")
def predict_churn(payload: ChurnPredictionInput):
    input_df = build_churn_features(payload.model_dump())

    pred = int(churn_model.predict(input_df)[0])

    if hasattr(churn_model, "predict_proba"):
        proba = float(churn_model.predict_proba(input_df)[0][1])
    else:
        proba = None

    return {
        "prediction": pred,
        "label": "High Risk" if pred == 1 else "Low Risk",
        "churn_probability": round(proba, 4) if proba is not None else None
    }
    
    
@app.get("/promo-trend")
def promo_trend():
    if "Date" not in customer_df.columns:
        return {"error": "Date column not found"}

    if "Promo" not in customer_df.columns or "Churn" not in customer_df.columns:
        return {"error": "Promo or Churn column not found"}

    df = customer_df.copy()

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])

    df["Promo"] = pd.to_numeric(df["Promo"], errors="coerce").fillna(0)
    df["Churn"] = pd.to_numeric(df["Churn"], errors="coerce").fillna(0)

    # aggregate by date
    trend = df.groupby(["Date", "Promo"])["Churn"].mean().reset_index()

    promo_trend = trend[trend["Promo"] > 0]
    non_promo_trend = trend[trend["Promo"] == 0]

    return {
        "dates": sorted(df["Date"].dt.strftime("%Y-%m-%d").unique().tolist()),
        "promo_churn": promo_trend.groupby("Date")["Churn"].mean().fillna(0).tolist(),
        "non_promo_churn": non_promo_trend.groupby("Date")["Churn"].mean().fillna(0).tolist(),
    }


# -----------------------------
# UPLOAD ANALYTICS API
# -----------------------------
@app.post("/analyze-upload")
def analyze_upload(file: UploadFile = File(...)):
    global latest_upload_analysis

    try:
        df = pd.read_csv(file.file)
    except Exception as e:
        return {"error": f"Could not read uploaded CSV: {str(e)}"}

    if df.empty:
        return {"error": "Uploaded file is empty."}

    df.columns = [col.strip() for col in df.columns]

    sales_synonyms = ["sales", "revenue", "total_sales", "amount", "sales_amount"]
    date_synonyms = ["date", "datetime", "timestamp", "day"]

    sales_col = next((col for col in df.columns if col.lower() in sales_synonyms), None)
    if not sales_col:
        return {"error": "Could not identify a Sales/Revenue column."}

    df = df.rename(columns={sales_col: "Sales"})
    df["Sales"] = pd.to_numeric(df["Sales"], errors="coerce")
    df = df.dropna(subset=["Sales"]).copy()

    date_col = next((col for col in df.columns if col.lower() in date_synonyms), None)
    if date_col:
        df = df.rename(columns={date_col: "Date"})
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    if "Store" not in df.columns:
        df["Store"] = 1
    if "Promo" not in df.columns:
        df["Promo"] = 0
    if "SchoolHoliday" not in df.columns:
        df["SchoolHoliday"] = 0
    if "StateHoliday" not in df.columns:
        df["StateHoliday"] = 0

    df = ensure_forecast_columns(df)

    kpis = {
        "Total_Rows": int(len(df)),
        "Total_Sales": round(float(df["Sales"].sum()), 2),
        "Avg_Sales": round(float(df["Sales"].mean()), 2),
    }

    if "Customers" in df.columns:
        kpis["Total_Customers"] = int(pd.to_numeric(df["Customers"], errors="coerce").fillna(0).sum())

    forecast = None
    warning = None

    missing_features = [f for f in forecast_features if f not in df.columns]

    if "Date" in df.columns:
        if len(missing_features) == 0:
            try:
                forecast = forecast_next_days(forecast_model, df, 7)
            except Exception as e:
                warning = f"Forecast failed: {str(e)}"
        else:
            warning = f"Forecast skipped. Missing: {', '.join(missing_features)}"
    else:
        warning = "Forecast skipped because no valid Date column was found."

    result = {
        "kpis": kpis,
        "forecast": forecast,
        "warning": warning
    }

    latest_upload_analysis = result
    return result


# -----------------------------
# CHAT TOOLS
# -----------------------------
def tool_get_store_dashboard(store_id: int) -> str:
    """Get KPI summary and 7-day forecast for a given store."""
    result = store_dashboard(store_id)
    return json.dumps(result, default=str)


def tool_get_customer_kpis(store_id: int) -> str:
    """Get customer KPI summary for a given store."""
    result = customer_kpis(store_id)
    return json.dumps(result, default=str)


def tool_get_churn_kpis(store_id: int) -> str:
    """Get churn KPI summary for a given store."""
    result = churn_kpis(store_id)
    return json.dumps(result, default=str)


def tool_get_promo_churn() -> str:
    """Get churn comparison for promo users versus non-promo users."""
    result = promo_churn()
    return json.dumps(result, default=str)


def tool_get_store_history(store_id: int) -> str:
    """Get the last 30 days of sales and customers for a given store."""
    result = store_history(store_id)
    return json.dumps(result, default=str)


def tool_get_uploaded_analysis() -> str:
    """Get the latest uploaded CSV analysis result."""
    if not latest_upload_analysis:
        return json.dumps({"message": "No uploaded analysis available yet."})
    return json.dumps(latest_upload_analysis, default=str)


# -----------------------------
# BUILD CHAT AGENT
# -----------------------------
if llm:
    chat_agent = create_agent(
        model=llm,
        tools=[
            tool_get_store_dashboard,
            tool_get_customer_kpis,
            tool_get_churn_kpis,
            tool_get_promo_churn,
            tool_get_store_history,
            tool_get_uploaded_analysis,
        ],
        system_prompt=(
            "You are a retail analytics assistant for a FastAPI dashboard. "
            "Use tools whenever live data is needed. "
            "If the user asks about a specific store, use that store id. "
            "If a store id is provided in context, prefer that store unless the user names another one. "
            "Be concise, clear, and business-friendly. "
            "Do not invent numbers."
        ),
    )


# -----------------------------
# CHAT API
# -----------------------------
@app.post("/chat")
def chat(request: ChatRequest):
    if chat_agent is None:
        return {
            "answer": "Chatbot is not configured yet. Please set NVIDIA_API_KEY in your environment."
        }

    context_parts = []
    if request.page:
        context_parts.append(f"Current page: {request.page}")
    if request.store_id is not None:
        context_parts.append(f"Current selected store id: {request.store_id}")

    user_message = request.message
    if context_parts:
        user_message = f"{' | '.join(context_parts)}\nUser question: {request.message}"

    try:
        response = chat_agent.invoke(
            {
                "messages": [
                    {"role": "user", "content": user_message}
                ]
            }
        )

        messages = response.get("messages", [])
        answer = "Sorry, I could not generate a response."

        for msg in reversed(messages):
            content = getattr(msg, "content", None)
            if isinstance(content, str) and content.strip():
                answer = content
                break

        return {"answer": answer}

    except Exception as e:
        return {"answer": f"Chat error: {str(e)}"}