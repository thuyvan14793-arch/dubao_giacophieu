# app.py
# Ung dung du bao gia dong cua co phieu VNM bang LSTM
# Cai dat: pip install streamlit yfinance pandas numpy scikit-learn tensorflow plotly

import random
from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential

# -------------------- Cau hinh --------------------
st.set_page_config(
    page_title="Dự báo giá cổ phiếu VNM",
    page_icon="📈",
    layout="wide"
)

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

st.title("📈 Dự báo giá cổ phiếu bằng mô hình LSTM")
st.caption("Dữ liệu lịch sử từ Yahoo Finance • Mã mặc định: VNM.VN")

# -------------------- Tham so tren sidebar --------------------
with st.sidebar:
    st.header("Thiết lập mô hình")
    ticker = st.text_input("Mã cổ phiếu Yahoo Finance", value="VNM.VN").strip().upper()
    start_date = st.date_input("Ngày bắt đầu lấy dữ liệu", value=date(2018, 1, 1))
    window_size = st.slider("Số phiên đầu vào (window)", min_value=20, max_value=120, value=60, step=5)
    forecast_days = st.slider("Số phiên cần dự báo", min_value=1, max_value=30, value=10)
    epochs = st.slider("Số epochs tối đa", min_value=5, max_value=50, value=10, step=5)
    train_button = st.button("Tải dữ liệu và chạy dự báo", type="primary", use_container_width=True)

st.info(
    "Mô hình dự báo giá đóng cửa dựa trên các phiên gần nhất. "
    "Kết quả chỉ phục vụ mục đích học tập, không phải khuyến nghị đầu tư."
)


@st.cache_data(ttl=3600, show_spinner=False)
def load_stock_data(symbol: str, start: date) -> pd.DataFrame:
    # end là ngày loại trừ, nên dùng ngày mai để bao gồm ngày hiện tại nếu có dữ liệu.
    end = date.today() + timedelta(days=1)
    df = yf.download(
        symbol,
        start=start.isoformat(),
        end=end.isoformat(),
        auto_adjust=False,
        progress=False
    )
    if df.empty:
        return pd.DataFrame()

    # yfinance có thể trả về cột MultiIndex tùy phiên bản.
    if isinstance(df.columns, pd.MultiIndex):
        if "Close" in df.columns.get_level_values(0):
            close = df.xs("Close", axis=1, level=0)
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
        else:
            return pd.DataFrame()
    else:
        if "Close" not in df.columns:
            return pd.DataFrame()
        close = df["Close"]

    result = pd.DataFrame({"Close": pd.to_numeric(close, errors="coerce")})
    result.index = pd.to_datetime(df.index)
    result = result.dropna()
    result.index.name = "Date"
    return result


def make_sequences(values: np.ndarray, window: int):
    X, y = [], []
    for idx in range(window, len(values)):
        X.append(values[idx - window:idx, 0])
        y.append(values[idx, 0])
    return np.asarray(X), np.asarray(y)


def build_model(window: int):
    model = Sequential([
        tf.keras.Input(shape=(window, 1)),
        LSTM(50, return_sequences=True),
        Dropout(0.1),
        LSTM(50),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


def run_forecast(df: pd.DataFrame, window: int, horizon: int, max_epochs: int):
    values = df[["Close"]].to_numpy(dtype=np.float32)
    if len(values) <= window + 20:
        raise ValueError(
            f"Chỉ có {len(values)} phiên dữ liệu. Hãy chọn khoảng thời gian dài hơn "
            f"để có nhiều hơn {window + 20} phiên."
        )

    # Chia theo thời gian: 80% đầu dùng huấn luyện, 20% cuối dùng kiểm tra.
    split_idx = int(len(values) * 0.8)
    train_raw = values[:split_idx]
    test_raw = values[split_idx:]

    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_raw)
    test_scaled = scaler.transform(test_raw)

    X_train, y_train = make_sequences(train_scaled, window)
    # Ghép phần cuối tập train để tạo đủ cửa sổ cho điểm test đầu tiên.
    test_context = np.vstack([train_scaled[-window:], test_scaled])
    X_test, y_test = make_sequences(test_context, window)

    X_train = X_train.reshape((-1, window, 1))
    X_test = X_test.reshape((-1, window, 1))

    # Tách validation theo thứ tự thời gian, không xáo trộn.
    val_size = max(1, int(len(X_train) * 0.1))
    if len(X_train) <= val_size:
        raise ValueError("Không đủ dữ liệu để chia tập huấn luyện và validation.")

    X_fit, y_fit = X_train[:-val_size], y_train[:-val_size]
    X_val, y_val = X_train[-val_size:], y_train[-val_size:]

    model = build_model(window)
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )
    history = model.fit(
        X_fit, y_fit,
        validation_data=(X_val, y_val),
        epochs=max_epochs,
        batch_size=32,
        shuffle=False,
        callbacks=[early_stop],
        verbose=0
    )

    test_pred_scaled = model.predict(X_test, verbose=0)
    test_pred = scaler.inverse_transform(test_pred_scaled).ravel()
    y_test_real = scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()

    rmse = float(np.sqrt(mean_squared_error(y_test_real, test_pred)))
    mae = float(mean_absolute_error(y_test_real, test_pred))
    r2 = float(r2_score(y_test_real, test_pred)) if len(y_test_real) > 1 else float("nan")

    # Dự báo cuốn chiếu nhiều phiên: mỗi dự báo được đưa lại vào cửa sổ kế tiếp.
    current_window = scaler.transform(values)[-window:].reshape(1, window, 1)
    future_scaled = []
    for _ in range(horizon):
        next_value = model.predict(current_window, verbose=0)[0, 0]
        future_scaled.append(next_value)
        next_step = np.array(next_value, dtype=np.float32).reshape(1, 1, 1)
        current_window = np.concatenate([current_window[:, 1:, :], next_step], axis=1)

    future_prices = scaler.inverse_transform(
        np.asarray(future_scaled).reshape(-1, 1)
    ).ravel()

    test_dates = df.index[split_idx:]
    future_dates = pd.bdate_range(
        start=df.index[-1] + pd.offsets.BDay(1),
        periods=horizon
    )

    return {
        "model": model,
        "history": history,
        "test_dates": test_dates,
        "actual": y_test_real,
        "predicted": test_pred,
        "future_dates": future_dates,
        "future_prices": future_prices,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "train_count": len(X_fit),
        "test_count": len(X_test),
    }


if train_button:
    st.cache_data.clear()
    st.session_state.pop("forecast_result", None)

    if not ticker:
        st.error("Vui lòng nhập mã cổ phiếu.")
    else:
        with st.spinner(f"Đang tải dữ liệu {ticker} và huấn luyện mô hình LSTM..."):
            try:
                stock_df = load_stock_data(ticker, start_date)
                if stock_df.empty:
                    st.error("Không tìm thấy dữ liệu. Hãy kiểm tra mã cổ phiếu Yahoo Finance.")
                else:
                    result = run_forecast(stock_df, window_size, forecast_days, epochs)
                    st.session_state["stock_df"] = stock_df
                    st.session_state["forecast_result"] = result
                    st.session_state["ticker"] = ticker
            except Exception as exc:
                st.error(f"Không thể hoàn tất dự báo: {exc}")

if "forecast_result" in st.session_state:
    df = st.session_state["stock_df"]
    result = st.session_state["forecast_result"]
    active_ticker = st.session_state["ticker"]

    st.subheader(f"Tổng quan dữ liệu: {active_ticker}")
    left, middle, right = st.columns(3)
    left.metric("Giá đóng cửa gần nhất", f"{df['Close'].iloc[-1]:,.2f}")
    middle.metric("Số phiên dữ liệu", f"{len(df):,}")
    right.metric("Số epochs đã chạy", f"{len(result['history'].history['loss'])}")

    tab_price, tab_loss, tab_future, tab_data = st.tabs([
        "📉 Thực tế và dự đoán",
        "🧠 Quá trình huấn luyện",
        "🔮 Dự báo tương lai",
        "📋 Dữ liệu lịch sử"
    ])

    with tab_price:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=result["test_dates"], y=result["actual"],
            mode="lines", name="Giá thực tế",
            line=dict(color="#2563eb", width=2)
        ))
        fig.add_trace(go.Scatter(
            x=result["test_dates"], y=result["predicted"],
            mode="lines", name="Giá dự đoán",
            line=dict(color="#ea580c", width=2, dash="dash")
        ))
        fig.update_layout(
            title="Giá đóng cửa thực tế và giá dự đoán trên tập kiểm tra",
            xaxis_title="Ngày giao dịch",
            yaxis_title="Giá đóng cửa",
            hovermode="x unified",
            template="plotly_white",
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Chỉ số đánh giá trên tập kiểm tra")
        c1, c2, c3 = st.columns(3)
        c1.metric("RMSE", f"{result['rmse']:,.4f}")
        c2.metric("MAE", f"{result['mae']:,.4f}")
        c3.metric("R² Score", f"{result['r2']:.4f}")

    with tab_loss:
        loss_fig = go.Figure()
        epochs_x = list(range(1, len(result["history"].history["loss"]) + 1))
        loss_fig.add_trace(go.Scatter(
            x=epochs_x, y=result["history"].history["loss"],
            mode="lines+markers", name="Training Loss",
            line=dict(color="#2563eb", width=2)
        ))
        loss_fig.add_trace(go.Scatter(
            x=epochs_x, y=result["history"].history["val_loss"],
            mode="lines+markers", name="Validation Loss",
            line=dict(color="#dc2626", width=2)
        ))
        loss_fig.update_layout(
            title="Diễn biến hàm mất mát trong quá trình huấn luyện",
            xaxis_title="Epoch",
            yaxis_title="Mean Squared Error",
            hovermode="x unified",
            template="plotly_white",
            height=450
        )
        st.plotly_chart(loss_fig, use_container_width=True)

    with tab_future:
        future_df = pd.DataFrame({
            "Ngày dự kiến": result["future_dates"].strftime("%Y-%m-%d"),
            "Giá đóng cửa dự báo": result["future_prices"]
        })
        st.warning(
            "Các ngày dự báo là ngày làm việc dự kiến, chưa loại trừ ngày nghỉ lễ. "
            "Dự báo nhiều bước có thể sai lệch tích lũy."
        )

        future_fig = go.Figure()
        # Hiển thị một phần lịch sử gần nhất để dễ quan sát xu hướng.
        recent = df.tail(90)
        future_fig.add_trace(go.Scatter(
            x=recent.index, y=recent["Close"],
            mode="lines", name="Giá lịch sử",
            line=dict(color="#2563eb", width=2)
        ))
        future_fig.add_trace(go.Scatter(
            x=result["future_dates"], y=result["future_prices"],
            mode="lines+markers", name="Giá dự báo",
            line=dict(color="#e11d48", width=2, dash="dash")
        ))
        future_fig.update_layout(
            title=f"Dự báo {len(result['future_prices'])} phiên tiếp theo",
            xaxis_title="Ngày",
            yaxis_title="Giá đóng cửa",
            hovermode="x unified",
            template="plotly_white",
            height=480
        )
        st.plotly_chart(future_fig, use_container_width=True)
        st.dataframe(future_df, use_container_width=True, hide_index=True)
        st.download_button(
            "⬇️ Tải bảng dự báo CSV",
            data=future_df.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"du_bao_{active_ticker.replace('.', '_')}.csv",
            mime="text/csv"
        )

    with tab_data:
        st.dataframe(
            df.tail(200).sort_index(ascending=False),
            use_container_width=True
        )

else:
    st.write("Chọn tham số ở thanh bên trái, sau đó nhấn **Tải dữ liệu và chạy dự báo**.")
