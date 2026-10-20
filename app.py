with tab1:
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        y=y_test_actual,
        mode="lines",
        name="Giá thực tế",
        line=dict(color="royalblue", width=2)
    ))

    fig.add_trace(go.Scatter(
        y=y_predicted,
        mode="lines",
        name="Giá dự đoán",
        line=dict(color="darkorange", width=2, dash="dash")
    ))

    fig.update_layout(
        title="So sánh giá cổ phiếu thực tế và dự đoán",
        xaxis_title="Thời gian / Phiên giao dịch",
        yaxis_title="Giá cổ phiếu",
        hovermode="x unified",
        template="plotly_white",
        height=480,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    st.plotly_chart(fig, use_container_width=True)

with tab2:
    fig_loss = go.Figure()

    # Loss trên tập huấn luyện
    fig_loss.add_trace(go.Scatter(
        x=list(range(1, len(history.history["loss"]) + 1)),
        y=history.history["loss"],
        mode="lines",
        name="Training Loss",
        line=dict(width=2, color="seagreen")
    ))

    # Loss trên tập kiểm định
    fig_loss.add_trace(go.Scatter(
        x=list(range(1, len(history.history["val_loss"]) + 1)),
        y=history.history["val_loss"],
        mode="lines",
        name="Validation Loss",
        line=dict(width=2, color="crimson", dash="dot")
    ))

    fig_loss.update_layout(
        title="Quá trình hội tụ của mô hình",
        xaxis_title="Số lần huấn luyện (Epoch)",
        yaxis_title="Mean Squared Error",
        template="plotly_white",
        hovermode="x unified",
        height=480
    )

    st.plotly_chart(fig_loss, use_container_width=True)


# KHỐI 4: TỔNG HỢP KẾT QUẢ ĐÁNH GIÁ

st.markdown("---")
st.subheader("📊 Hiệu suất dự báo của mô hình")

col_rmse, col_mae, col_r2 = st.columns(3)

with col_rmse:
    st.metric(
        label="RMSE",
        value=f"{rmse:.4f}",
        help="Sai số bình phương trung bình căn bậc hai"
    )

with col_mae:
    st.metric(
        label="MAE",
        value=f"{mae:.4f}",
        help="Sai số tuyệt đối trung bình"
    )

with col_r2:
    st.metric(
        label="R² Score",
        value=f"{r2:.4f}",
        help="Mức độ phù hợp của mô hình với dữ liệu"
    )
