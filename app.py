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
