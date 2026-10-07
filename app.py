# import streamlit as st
# import pandas as pd
# from utils import mask_pii
# from models import load_model

# # Page Configuration
# st.set_page_config(
#     page_title="Support Email Classifier & PII Masker",
#     page_icon="🛡️",
#     layout="wide"
# )

# # Load Classifier Model
# @st.cache_resource
# def get_model():
#     return load_model()

# model = get_model()

# # UI Header
# st.title("🛡️ Support Email Classifier & PII Masking System")
# st.markdown("Paste or type a customer support email below to test PII masking and ticket category classification.")

# # Input Form
# with st.form("email_form"):
#     user_input = st.text_area(
#         "Enter Customer Support Email Body:",
#         height=150,
#         placeholder="Subject: Account login issue...\nMy name is John Doe, email is john@example.com, and my phone is 9876543210."
#     )
#     submit_button = st.form_submit_button("Process Email")

# if submit_button:
#     if user_input.strip() == "":
#         st.warning("Please enter some text to process.")
#     else:
#         with st.spinner("Processing PII masking and classification..."):
#             # 1. Mask PII
#             mask_result = mask_pii(user_input)
#             masked_email = mask_result["masked_email"]
#             entities = mask_result["list_of_masked_entities"]
            
#             # 2. Classify Category
#             predicted_category = model.predict([user_input])[0]
            
#         # Display Results in Columns
#         st.success("Processing Complete!")
        
#         col1, col2 = st.columns(2)
        
#         with col1:
#             st.subheader("📋 Classification Result")
#             st.metric(label="Predicted Category", value=predicted_category)
            
#             st.subheader("🔒 Masked Email Output")
#             st.info(masked_email)
            
#         with col2:
#             st.subheader("🔍 Detected PII Entities")
#             if entities:
#                 st.json(entities)
#             else:
#                 st.write("No sensitive PII entities detected.")
                
#         # Optional: Raw JSON view matching evaluation structure
#         with st.expander("View Raw JSON Output Structure"):
#             json_output = {
#                 "input_email_body": user_input,
#                 "list_of_masked_entities": entities,
#                 "masked_email": masked_email,
#                 "category_of_the_email": predicted_category
#             }
#             st.json(json_output)



import streamlit as st
import pandas as pd
from utils import mask_pii
from models import load_model

# 1. Page Configuration
st.set_page_config(
    page_title="Support Email Classifier & PII Masker",
    page_icon="🛡️",
    layout="centered"
)

# 2. Load Model Pipeline
@st.cache_resource
def get_model():
    return load_model()

model = get_model()

# 3. Clean Centered Header
st.markdown("<h1 style='text-align: center; color: #1E40AF;'>🛡️ Support Email Classifier & PII Masker</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 1.1rem; color: #64748B; margin-bottom: 2rem;'>Automated Support Classification & Secure PII Redaction Pipeline.</p>", unsafe_allow_html=True)

# 4. Input Card (Using bordered container)
with st.container(border=True):
    st.markdown("#### 📥 Input Customer Email")
    email_input = st.text_area(
        label="Email Body",
        height=180,
        placeholder="Paste the raw support email content here (e.g., 'My name is John and my card is 1234-5678...')",
        label_visibility="collapsed" # Hides the default label for a cleaner look
    )
    
    submit = st.button("✨ Analyze & Redact", type="primary", use_container_width=True)

# 5. Pipeline Execution & Output
if submit:
    if not email_input.strip():
        st.warning("⚠️ Please enter an email body to process.")
    else:
        # Sleek animated status box
        with st.status("Executing Support AI Pipeline...", expanded=True) as status:
            st.write("🔍 Scanning text via Regex & spaCy...")
            mask_result = mask_pii(email_input)
            masked_email = mask_result["masked_email"]
            entities = mask_result["list_of_masked_entities"]
            
            st.write("🧠 Classifying ticket intent via TF-IDF Logistic Regression...")
            predicted_category = model.predict([email_input])[0]
            
            status.update(label="Analysis Complete!", state="complete", expanded=False)
            
        # 6. Results Section
        st.markdown("<br>", unsafe_allow_html=True) # visual spacing
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="🎫 Predicted Ticket Category", value=predicted_category)
        with col2:
            st.metric(label="🔒 Sensitive Entities Redacted", value=len(entities))

        # Output Cards using Tabs
        with st.container(border=True):
            tab1, tab2, tab3 = st.tabs(["💬 Sanitized Email", "📋 Entity Report", "💻 JSON Payload"])
            
            # Tab 1: The masked email output
            with tab1:
                st.markdown("##### Processed Text (Safe for Database/LLM)")
                st.info(masked_email)
                
            # Tab 2: Clean data table of found entities
            with tab2:
                st.markdown("##### Detected PII / PCI Information")
                if entities:
                    df_entities = pd.DataFrame([
                        {
                            "Entity Type": e["classification"],
                            "Original Text": e["entity"],
                            "Start Pos": e["position"][0],
                            "End Pos": e["position"][1]
                        }
                        for e in entities
                    ])
                    st.dataframe(df_entities, use_container_width=True, hide_index=True)
                else:
                    st.success("✅ No PII or PCI data detected in this email.")
                    
            # Tab 3: The strict JSON requested by the assignment evaluation
            with tab3:
                st.markdown("##### Evaluator API Schema")
                st.json({
                    "input_email_body": email_input,
                    "list_of_masked_entities": entities,
                    "masked_email": masked_email,
                    "category_of_the_email": predicted_category
                })