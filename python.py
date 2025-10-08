import streamlit as st
import pandas as pd
import numpy as np
import json
from google import genai
from google.genai.errors import APIError
from io import StringIO
import re

# --- Cấu hình Trang Streamlit ---
st.set_page_config(
    page_title="App Đánh Giá Phương Án Kinh Doanh (Capital Budgeting)",
    layout="wide"
)

st.title("Ứng dụng Đánh giá Phương án Kinh doanh 📈")
st.markdown("Sử dụng Gemini AI để trích xuất dữ liệu và tính toán chỉ số hiệu quả dự án.")

# Thêm numpy để tính toán tài chính
# --- Cấu trúc dữ liệu yêu cầu từ AI (JSON Schema) ---
EXTRACTION_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "Vốn đầu tư (triệu)": {"type": "NUMBER", "description": "Tổng vốn đầu tư ban đầu của dự án (tại thời điểm 0), đơn vị triệu VND."},
        "Vòng đời dự án (năm)": {"type": "NUMBER", "description": "Thời gian hoạt động của dự án (số năm)."},
        "Doanh thu hàng năm (triệu)": {"type": "NUMBER", "description": "Doanh thu hoạt động hàng năm ước tính, đơn vị triệu VND. Giả định là con số trung bình hàng năm."},
        "Chi phí hàng năm (triệu)": {"type": "NUMBER", "description": "Tổng chi phí hoạt động hàng năm (chưa bao gồm thuế, lãi vay, khấu hao), đơn vị triệu VND. Giả định là con số trung bình hàng năm."},
        "WACC (%)": {"type": "NUMBER", "description": "Tỷ lệ chiết khấu WACC (Weighted Average Cost of Capital) của dự án, đơn vị phần trăm (%)."},
        "Thuế (%)": {"type": "NUMBER", "description": "Thuế suất thuế thu nhập doanh nghiệp, đơn vị phần trăm (%)."}
    },
    "required": [
        "Vốn đầu tư (triệu)", "Vòng đời dự án (năm)", 
        "Doanh thu hàng năm (triệu)", "Chi phí hàng năm (triệu)", 
        "WACC (%)", "Thuế (%)"
    ]
}

# --- Hàm gọi API Gemini để Trích xuất Dữ liệu Cấu trúc ---
def extract_data_from_document(document_text, api_key):
    """Sử dụng Gemini để trích xuất dữ liệu tài chính theo định dạng JSON."""
    try:
        client = genai.Client(api_key=api_key)
        
        # System prompt hướng dẫn AI đóng vai trò chuyên gia phân tích và tuân thủ JSON
        system_prompt = f"""
        Bạn là một chuyên gia phân tích tài chính. Nhiệm vụ của bạn là đọc bản tóm tắt phương án kinh doanh do khách hàng cung cấp và trích xuất sáu (6) thông tin tài chính cốt lõi sau: Vốn đầu tư, Vòng đời dự án, Doanh thu hàng năm, Chi phí hàng năm, WACC, và Thuế suất.
        
        Cần lưu ý:
        1. Đơn vị tiền tệ là triệu VND.
        2. Doanh thu và Chi phí là các con số ổn định hàng năm (Average Annual Figures).
        3. WACC và Thuế suất phải là giá trị phần trăm (%).
        4. Kết quả TRẢ VỀ DUY NHẤT dưới dạng JSON theo schema đã định nghĩa. TUYỆT ĐỐI KHÔNG THÊM BẤT KỲ VĂN BẢN GIẢI THÍCH NÀO NGOÀI KHỐI JSON.
        """

        prompt = f"""
        Dưới đây là nội dung văn bản của phương án kinh doanh:
        ---
        {document_text}
        ---
        Hãy trích xuất 6 thông tin tài chính bắt buộc và trả về dưới định dạng JSON.
        """

        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            system_instruction=system_prompt,
            config={
                "response_mime_type": "application/json",
                "response_schema": EXTRACTION_SCHEMA
            }
        )
        
        # Kiểm tra và trả về JSON
        if response.text:
            return json.loads(response.text)
        return None

    except APIError as e:
        st.error(f"Lỗi gọi Gemini API: Vui lòng kiểm tra Khóa API hoặc giới hạn sử dụng. Chi tiết lỗi: {e}")
        return None
    except json.JSONDecodeError:
        st.error("Lỗi giải mã JSON: AI không trả về đúng định dạng JSON. Vui lòng kiểm tra nội dung file.")
        return None
    except Exception as e:
        st.error(f"Đã xảy ra lỗi không xác định trong quá trình trích xuất dữ liệu: {e}")
        return None

# --- Hàm tính toán Dòng tiền và các Chỉ số ---
@st.cache_data
def calculate_project_metrics(data):
    """Xây dựng bảng dòng tiền và tính toán các chỉ số NPV, IRR, PP, DPP."""
    
    # 1. Trích xuất thông số
    I0 = data.get('Vốn đầu tư (triệu)')
    N = int(data.get('Vòng đời dự án (năm)'))
    Revenue = data.get('Doanh thu hàng năm (triệu)')
    Cost = data.get('Chi phí hàng năm (triệu)')
    WACC = data.get('WACC (%)') / 100.0  # Chuyển WACC sang số thập phân
    Tax = data.get('Thuế (%)') / 100.0    # Chuyển Thuế sang số thập phân
    
    # Giả định: Không có khấu hao và Giá trị thanh lý (Simplified OCF)
    
    # 2. Tính toán Dòng tiền hoạt động hàng năm (Annual OCF)
    # OCF = (Revenue - Cost) * (1 - Tax)
    Annual_OCF = (Revenue - Cost) * (1 - Tax)
    
    if Annual_OCF <= 0:
        raise ValueError("Lợi nhuận sau thuế hàng năm không dương. Dự án không khả thi.")

    # 3. Xây dựng Bảng dòng tiền
    years = list(range(N + 1))
    cash_flows = [-I0] + [Annual_OCF] * N
    
    df_cf = pd.DataFrame({
        'Năm': years,
        'Dòng tiền (CF)': cash_flows,
        'Hệ số chiết khấu': [1.0] + [1 / (1 + WACC)**t for t in range(1, N + 1)],
    })
    
    # Dòng tiền chiết khấu
    df_cf['Dòng tiền chiết khấu (DCF)'] = df_cf['Dòng tiền (CF)'] * df_cf['Hệ số chiết khấu']
    
    # Dòng tiền tích lũy và Dòng tiền chiết khấu tích lũy
    df_cf['CF Tích lũy'] = df_cf['Dòng tiền (CF)'].cumsum()
    df_cf['DCF Tích lũy'] = df_cf['Dòng tiền chiết khấu (DCF)'].cumsum()
    
    # 4. Tính toán các chỉ số
    
    # 4a. NPV (Net Present Value)
    NPV = df_cf['Dòng tiền chiết khấu (DCF)'].sum()
    
    # 4b. IRR (Internal Rate of Return) - Sử dụng numpy.irr
    # Lưu ý: IRR có thể không tính được nếu dòng tiền không có sự thay đổi dấu (từ âm sang dương)
    try:
        IRR = np.irr(cash_flows)
    except:
        IRR = np.nan # Gán NaN nếu numpy không thể tính được IRR
        
    # 4c. PP (Payback Period - Thời gian hoàn vốn không chiết khấu)
    # Tìm năm đầu tiên mà CF Tích lũy >= 0
    df_positive_cf = df_cf[df_cf['CF Tích lũy'] >= 0]
    if not df_positive_cf.empty:
        payback_year_int = df_positive_cf.iloc[0]['Năm']
        # Tính chiết khấu phần lẻ (interpolation)
        if payback_year_int > 0:
            cf_before = df_cf.loc[payback_year_int - 1, 'CF Tích lũy']
            cf_current = df_cf.loc[payback_year_int, 'Dòng tiền (CF)']
            PP = (payback_year_int - 1) + (abs(df_cf.loc[0, 'Dòng tiền (CF)']) - df_cf.loc[payback_year_int - 1, 'CF Tích lũy']) / cf_current
        else: # Hoàn vốn ngay trong năm đầu tiên
            PP = (abs(df_cf.loc[0, 'Dòng tiền (CF)']) / Annual_OCF)
    else:
        PP = N + 1 # Không hoàn vốn trong vòng đời dự án

    # 4d. DPP (Discounted Payback Period - Thời gian hoàn vốn có chiết khấu)
    # Tìm năm đầu tiên mà DCF Tích lũy >= 0
    df_positive_dcf = df_cf[df_cf['DCF Tích lũy'] >= 0]
    if not df_positive_dcf.empty:
        dpp_year_int = df_positive_dcf.iloc[0]['Năm']
        # Tính chiết khấu phần lẻ (interpolation)
        if dpp_year_int > 0:
            dcf_before = df_cf.loc[dpp_year_int - 1, 'DCF Tích lũy']
            dcf_current = df_cf.loc[dpp_year_int, 'Dòng tiền chiết khấu (DCF)']
            # Lấy CF gốc I0 = abs(df_cf.loc[0, 'Dòng tiền (CF)'])
            DPP = (dpp_year_int - 1) + (abs(df_cf.loc[0, 'Dòng tiền (CF)']) - df_cf.loc[dpp_year_int - 1, 'DCF Tích lũy']) / dcf_current
        else: # Hoàn vốn ngay trong năm đầu tiên
             # Điều này rất khó xảy ra, nhưng xử lý để tránh lỗi
             DPP = (abs(df_cf.loc[0, 'Dòng tiền (CF)']) / (Annual_OCF * df_cf.loc[1, 'Hệ số chiết khấu']))
    else:
        DPP = N + 1 # Không hoàn vốn trong vòng đời dự án
        
    metrics = {
        'NPV': NPV,
        'IRR': IRR,
        'PP': PP,
        'DPP': DPP
    }
    
    return df_cf, metrics, data

# --- Hàm gọi API Gemini để Phân tích Chỉ số ---
def get_ai_analysis_metrics(data, metrics, api_key):
    """Gửi các chỉ số đã tính toán đến Gemini AI để nhận phân tích."""
    try:
        client = genai.Client(api_key=api_key)
        
        # Format lại dữ liệu cho prompt
        formatted_metrics = (
            f"Vốn đầu tư: {data['Vốn đầu tư (triệu)']:,.0f} triệu VND, "
            f"Vòng đời: {data['Vòng đời dự án (năm)']} năm, "
            f"WACC (Tỷ lệ chiết khấu): {data['WACC (%)']:.2f}%, "
            f"Lợi nhuận sau thuế hàng năm (Ước tính): {((data['Doanh thu hàng năm (triệu)'] - data['Chi phí hàng năm (triệu)']) * (1 - data['Thuế (%)']/100)):,.0f} triệu VND\n"
            f"--- Chỉ số đánh giá ---\n"
            f"NPV (Giá trị hiện tại ròng): {metrics['NPV']:,.0f} triệu VND\n"
            f"IRR (Tỷ suất sinh lời nội bộ): {metrics['IRR'] * 100:.2f}%\n"
            f"PP (Thời gian hoàn vốn): {metrics['PP']:.2f} năm\n"
            f"DPP (Thời gian hoàn vốn chiết khấu): {metrics['DPP']:.2f} năm"
        )
        
        # System prompt hướng dẫn AI đóng vai trò chuyên gia đánh giá dự án
        system_prompt = "Bạn là một chuyên gia thẩm định dự án đầu tư. Hãy phân tích khách quan và chuyên sâu về tính khả thi của dự án dựa trên các chỉ số NPV, IRR, PP, và DPP. Đưa ra kết luận (Accept/Reject) rõ ràng. Bài phân tích nên chia thành 3 đoạn: 1. Đánh giá NPV & IRR so với WACC. 2. Phân tích khả năng thanh khoản (PP, DPP) và rủi ro. 3. Kết luận tổng thể."

        prompt = f"""
        Phân tích tính khả thi của dự án kinh doanh với các thông số và chỉ số đã tính toán sau:
        
        {formatted_metrics}
        """

        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            system_instruction=system_prompt,
            tools=[{"google_search": {}}] # Có thể sử dụng Google Search để so sánh với thị trường
        )
        
        return response.text

    except APIError as e:
        return f"Lỗi gọi Gemini API: Vui lòng kiểm tra Khóa API hoặc giới hạn sử dụng. Chi tiết lỗi: {e}"
    except Exception as e:
        return f"Đã xảy ra lỗi không xác định: {e}"

# --- Chức năng 1: Tải File và Trích xuất Dữ liệu ---

# Tải file/Nội dung
st.subheader("1. Tải File hoặc Dán Nội dung Phương án Kinh doanh")

uploaded_file = st.file_uploader(
    "Tải lên file (.txt, .md, .docx). Tốt nhất nên dùng .txt hoặc .md để AI dễ đọc nội dung.",
    type=['txt', 'md', 'docx']
)

# Area để dán nội dung từ Word (giải pháp thay thế cho thư viện docx)
document_text_area = st.text_area(
    "Hoặc dán toàn bộ nội dung file Word vào đây:",
    height=200,
    key="document_text_input"
)

# Xử lý nội dung đầu vào
if uploaded_file is not None:
    try:
        if uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            st.warning("Đã phát hiện file Word (.docx). Vì giới hạn thư viện, vui lòng ĐẢM BẢO nội dung file Word được dán vào ô bên dưới để AI có thể đọc chính xác.")
            # Đọc nội dung file Word thô, nhưng vẫn ưu tiên text_area
            # Dù sao, ta sẽ cố gắng đọc file text nếu đó là file text
            file_contents = StringIO(uploaded_file.getvalue().decode("utf-8")).read()
            if not document_text_area:
                 document_text = file_contents
            else:
                 document_text = document_text_area # Ưu tiên nội dung dán
        else:
            stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
            document_text = stringio.read()
    except UnicodeDecodeError:
        st.error("Lỗi: Không thể giải mã file. Vui lòng đảm bảo file là văn bản thuần túy (.txt, .md) hoặc dán nội dung vào ô bên dưới.")
        document_text = ""
else:
    document_text = document_text_area

# Nút thực hiện thao tác lọc dữ liệu
if st.button("Lọc Dữ liệu Tài chính bằng AI", type="primary", use_container_width=True) and document_text:
    
    api_key = st.secrets.get("GEMINI_API_KEY") 
    
    if not api_key:
        st.error("Lỗi: Không tìm thấy Khóa API 'GEMINI_API_KEY'. Vui lòng cấu hình Khóa trong Streamlit Secrets.")
    else:
        with st.spinner('Đang gửi nội dung và chờ Gemini AI trích xuất dữ liệu có cấu trúc...'):
            extracted_data = extract_data_from_document(document_text, api_key)
            
            if extracted_data:
                st.session_state['extracted_data'] = extracted_data
                st.success("Trích xuất dữ liệu thành công!")
            else:
                st.error("Không thể trích xuất dữ liệu. Vui lòng kiểm tra lại nội dung file.")

# --- Hiển thị và Tính toán sau khi trích xuất ---

if 'extracted_data' in st.session_state:
    data = st.session_state['extracted_data']
    
    try:
        # Tính toán các chỉ số
        df_cf, metrics, clean_data = calculate_project_metrics(data)

        st.divider()

        # Hiển thị dữ liệu đã trích xuất
        st.subheader("2. Dữ liệu Tài chính đã Lọc")
        col_list = list(clean_data.items())
        
        # Hiển thị trong 3 cột
        cols = st.columns(3)
        for i, (key, value) in enumerate(col_list):
            cols[i % 3].metric(
                label=key,
                value=f"{value:,.2f}" if isinstance(value, (int, float)) else str(value),
                help="Đơn vị tiền tệ là Triệu VND, WACC và Thuế là %"
            )

        st.divider()

        # Hiển thị Bảng Dòng tiền
        st.subheader("3. Bảng Dòng tiền Dự án (Cash Flow Table)")
        
        # Định dạng cột tiền tệ
        st.dataframe(df_cf.style.format({
            'Dòng tiền (CF)': '{:,.0f}',
            'Hệ số chiết khấu': '{:.4f}',
            'Dòng tiền chiết khấu (DCF)': '{:,.0f}',
            'CF Tích lũy': '{:,.0f}',
            'DCF Tích lũy': '{:,.0f}',
        }), use_container_width=True, hide_index=True)
        
        st.divider()
        
        # Hiển thị các chỉ số đánh giá
        st.subheader("4. Các Chỉ số Đánh giá Hiệu quả Dự án")
        
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        
        with col_m1:
            st.metric("NPV (Giá trị hiện tại ròng)", f"{metrics['NPV']:,.0f} Triệu VND")
        
        with col_m2:
            st.metric("IRR (Tỷ suất sinh lời nội bộ)", f"{metrics['IRR'] * 100:.2f}%")
        
        with col_m3:
            st.metric("PP (Thời gian hoàn vốn)", f"{metrics['PP']:.2f} Năm")
        
        with col_m4:
            st.metric("DPP (Thời gian hoàn vốn CĐ)", f"{metrics['DPP']:.2f} Năm")
            
        st.divider()

        # Chức năng Yêu cầu AI Phân tích
        st.subheader("5. Phân tích Hiệu quả Dự án (AI)")
        
        if st.button("Yêu cầu AI Phân tích Chỉ số", use_container_width=True):
            api_key = st.secrets.get("GEMINI_API_KEY") 
            if api_key:
                with st.spinner('Đang gửi dữ liệu và chờ Gemini phân tích...'):
                    ai_result = get_ai_analysis_metrics(clean_data, metrics, api_key)
                    st.markdown("---")
                    st.markdown("**Kết quả Phân tích từ Gemini AI:**")
                    st.info(ai_result)
            else:
                 st.error("Lỗi: Không tìm thấy Khóa API. Vui lòng cấu hình Khóa 'GEMINI_API_KEY' trong Streamlit Secrets.")

    except ValueError as ve:
        st.error(f"Lỗi tính toán: {ve}")
    except Exception as e:
        st.error(f"Có lỗi xảy ra trong quá trình tính toán: {e}")

else:
    st.info("Vui lòng tải lên hoặc dán nội dung file và nhấn nút 'Lọc Dữ liệu Tài chính bằng AI' để bắt đầu.")
