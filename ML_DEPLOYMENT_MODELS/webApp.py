import random

import pandas as pd
import streamlit as st
import sklearn
import base64
import time
import pickle
from PIL import Image
import urllib.request
st.set_page_config(page_title="BANDORA LOAN APPROVAL WEBAPP", page_icon="random", layout="wide", initial_sidebar_state="expanded")

def add_bg_from_local(image_file):
    path=image_file
    with open(path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
        st.markdown(
             f"""
             <style>.stApp {{background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
             background-size: cover}}             </style>""", unsafe_allow_html=True
        )


add_bg_from_local("bckg.jpg")

classifier_pipeline = pickle.load(open('classifier_gboost_pipeline.pkl', 'rb'))


Regressor_pipeline = pickle.load(open('regression_xgboost_pipeline.pkl', 'rb'))

@st.experimental_memo
def create_input_Dataframe():
    input_dictionary = {
        "LanguageCode": LanguageCode,
        "HomeOwnershipType": HomeOwnershipType,
        "Restructured": Restructured,
        "IncomeTotal": IncomeTotal,
        "LiabilitiesTotal": LiabilitiesTotal,
        "LoanDuration": LoanDuration,
        "AppliedAmount": AppliedAmount,
        "Amount": Amount,
        "Interest": Interest,
        "EMI": EMI,
        "PreviousRepaymentsBeforeLoan": PreviousRepaymentsBeforeLoan,
        "MonthlyPaymentDay": MonthlyPaymentDay,

        "PrincipalPaymentsMade": PrincipalPaymentsMade,
        "InterestAndPenaltyPaymentsMade": InterestAndPenaltyPaymentsMade,
        "PrincipalBalance": PrincipalBalance,
        "InterestAndPenaltyBalance": InterestAndPenaltyBalance,
        "Bids": BidsPortfolioManager + BidsApi,
        "Rating": Rating
    }

    DF = pd.DataFrame(input_dictionary, index=[0])
    return DF
def Classifier():
    global result
    inputs = create_input_Dataframe()
    prediction = classifier_pipeline.predict(inputs)
    if prediction == 1:
        result = "Defaulter"
    if prediction == 0:
        result = "Not Defaulter"
    return result

def default():
    d = Classifier()
    if d == 1:
        defaults = 1
    else:
        defaults = 0
    return defaults


@st.experimental_memo
def create_regression_input():
    regr_dict={
        # NUM data
        "BidsPortfolioManager": BidsPortfolioManager,
        "BidsApi": BidsApi,
        "BidsManual": BidsManual,
        "Age": Age,
        "AppliedAmount": AppliedAmount,
        "Amount": Amount,
        "Interest": Interest,
        "LoanDuration": LoanDuration,
        "MonthlyPayment": MonthlyPayment,
        "EmploymentDurationCurrentEmployer": EmploymentDurationCurrentEmployer,
        "IncomeTotal": IncomeTotal,
        "ExistingLiabilities": ExistingLiabilities,
        "LiabilitiesTotal": LiabilitiesTotal,
        "RefinanceLiabilities": RefinanceLiabilities,
        "DebtToIncome": DebtToIncome,
        "FreeCash": Freecash,
        "MonthlyPaymentDay": MonthlyPaymentDay,
        "CreditScoreEeMini'": CreditScoreEeMini,
        "PrincipalPaymentsMade": PrincipalPaymentsMade,
        "InterestAndPenaltyPaymentsMade": InterestAndPenaltyPaymentsMade,
        "PrincipalBalance": PrincipalBalance,
        "InterestAndPenaltyBalance": InterestAndPenaltyBalance,
        "NoOfPreviousLoansBeforeLoan": NoOfPreviousLoansBeforeLoan,
        "AmountOfPreviousLoansBeforeLoan": AmountOfPreviousLoansbeforeloan,
        "PreviousEarlyRepaymentsCountBeforeLoan": PreviousEarlyRepaymentsCountBeforeLoan,
        # cat
        "NewCreditCustomer": NewCreditCustomer,
        "VerificationType": VerificationType,
        "LanguageCode": LanguageCode,
        "Gender": Gender,
        "UseOfLoan": UseOfLoan,
        "Education": Education,
        "MaritalStatus": MaritalStatus,
        "EmploymentStatus": EmploymentStatus,
        "OccupationArea": OccupationArea,
        "HomeOwnershipType": HomeOwnershipType,
        "RecoveryStage": RecoveryStage,
        "Rating": Rating,
        "Restructured": Restructured,
        "CreditScoreEsMicroL": CreditScoreEsMicroL,
        "Default": Default


    }
    DF_reg = pd.DataFrame(regr_dict, index=[0])
    return DF_reg

# regression model using
def Regressor():
    inputs = create_regression_input()
    prediction = Regressor_pipeline.predict(inputs)
    Result=pd.DataFrame({"EMI": 0, "ELA": 0, "ROI": 0})
    Result['EMI']=prediction[0]
    Result['ELA']=prediction[1]
    Result['ROI']=prediction[2]
    return Result


# getting user filled data
@st.experimental_memo
def load_data():
    dfs=create_regression_input()
    return dfs


# APP LAYOUT
st.title('Bandora Loan Approval Dashboard')
st.header("Borrower's Information")
#op1 = st.button(label="Personal Details")
st.subheader('Personal Background')
EmploymentDurationCurrentEmployer = st.selectbox('EmploymentDurationCurrentEmployer', (
"MoreThan5Years", "UpTo1Year", "UpTo5Years", "UpTo3Years", "UpTo4Years", "Other", "TrialPeriod"))
LanguageCode = st.selectbox('Languagecode', ("estonia", "Finish", "spanish", "other"))
HomeOwnershipType = st.selectbox('Home Ownership Type', (
"homeless", "Owner", "other", "Tenant_pre-furnished property", "Living with parents", "Mortgage",
"Tenant_unfurnished property", "other", "Joint ownership", "Joint tenant", "Council house", "Owner with encumbrance"))
Gender = st.selectbox('Gender', ("Male", "Woman", "Undefined"))
Education = st.selectbox('Education', (
"Basic education", "Primary education", "Vocational education", "Higher education", "other", "Secondary education"))
MaritalStatus = st.selectbox('MaritalStatus', ("Married", "Cohabitant", "Single", "Divorced", "Widow", "other"))
IncomeTotal = st.number_input('Total Income')
LiabilitiesTotal = st.number_input('Total Liabilities')
EmploymentStatus = st.selectbox('EmploymentStatus', ("Fully employed", "Self-employed_Entrepreneur_Retiree",
                                                     "Unemployed_Partially employed")
                                )
OccupationArea = st.selectbox('OccupationArea', ("Other", "Mining", "Processing", "Energy", "Utilities", "Construction",
                                                 "Retail and wholesale", "Transport and warehousing",
                                                 "Hospitality and catering", "Info and telecom",
                                                 "Finance and insurance",
                                                 "Real-estate", "Research", "Administrative",
                                                 "Civil service & military", "Education", "Healthcare and social help",
                                                 "Art and entertainment",
                                                 "Agriculture,forestry and fishing")
                              )
RefinanceLiabilities = st.number_input('Refinance Liabilities')
Freecash = st.number_input('Free Cash')
ExistingLiabilities = st.number_input('Existing Liabilities')
DebtToIncome = st.number_input('DebtToIncome')
Age = st.number_input('Age')

#op2 = st.button(label="LoanDetails"
st.subheader('Loan Details')
NewCreditCustomer = st.selectbox('NewCreditCustomer', ("True", "False"))
LoanDuration = st.number_input('Loan Duration (in months)')
AppliedAmount = st.number_input('Applied Loan Amount')
Amount = st.number_input('Amount (granted)')
Interest = st.number_input('Interest')
EMI = st.number_input('Equated Monthly Installment')
RecoveryStage = st.selectbox('RecoveryStage', ("Collection", "Recovery"))
UseOfLoan = st.selectbox('UseOfLoan', (
"other", "Home improvement", "Loan consolidation", "Vehicle", "Travel", "Business", "Education", "Any"))
VerificationType = st.selectbox('VerificationType',
                                ("Not set", "Income unverified", "Income unverified cross-referenced by phone",
                                 "Income verified", "Income and expenses verified"))
CreditScoreEsMicroL = st.selectbox('CreditScoreEsMicroL',
                                   ("M", "M3", "M5", "M1", "M9", "M2", "M6", "M4", "M8", "M7", "M10"))
NoOfPreviousLoansBeforeLoan = st.number_input('No Of Previous Loans Before Loan')
AmountOfPreviousLoansbeforeloan = st.text_input('Amount Of Previous Loans before loan')
CreditScoreEeMini = st.number_input('Credit Score Ee Mini')
PreviousEarlyRepaymentsCountBeforeLoan = st.number_input('PreviousEarlyRepaymentsCountBeforeLoan')
Restructured = st.selectbox('Restructured', ("False", "True"))

#op3 = st.button(label="PaymentofPreviousLoan")
st.subheader('Payment Details')
PreviousRepaymentsBeforeLoan = st.number_input('PreviousRepaymentsBeforeLoan')
MonthlyPaymentDay = st.number_input('MonthlyPaymentDay (digit)')
MonthlyPayment = st.number_input('Monthly Payment')
PrincipalPaymentsMade = st.number_input('Principal Payments Made')
InterestAndPenaltyPaymentsMade = st.number_input('Interest and Penalty Payments Made')

#op4=st.button(label="YourBalance")
st.subheader('Balance Details')
PrincipalBalance = st.number_input('PrincipalBalance')
InterestAndPenaltyBalance = st.number_input('InterestAndPenaltyBalance')
st.subheader('Amount of Investment offers made via')
BidsPortfolioManager = st.number_input('Bids through PortfolioManger')
BidsApi = st.number_input('Bids using Api')
BidsManual = st.number_input('Bids by Manual')

st.subheader('Other')
Rating = st.selectbox('Rating', ("A", "AA", "B", "C", "D", "E", "F", "HR"))


def random_emoji():
    st.session_state.emoji = random.choice(emojis)


# initialize emoji as a Session State variable
if "emoji" not in st.session_state:
    st.session_state.emoji = "👈"

emojis = ["🐶", "🐱", "🐭", "🐹", "🐰", "🦊", "🐻", "🐼"]
Default = default()
# if st.button(f"Check Status {st.session_state.emoji}", on_click=random_emoji):
#     Default = default()
#     st.header('Loan Application Status')
#     with st.spinner('Analyzing the Provided Information ...'):
#         time.sleep(5)
#     result = Classifier()
#     st.spinner(text="Analyzing the Information")
#
#     if result == "Defaulter":
#         st.write("Based on details provided, the user may default so loan is not approved, Thanks!")
#         time.sleep(3)
#         with st.spinner('Predicting preferred Loan details ...'):
#             time.sleep(5)
#             result=Regressor()
#             st.dataframe(result, use_container_width=st.session_state.use_container_width)
            #st.balloons()

    # elif result == "Not Defaulter":
    #     st.write("Congratulations! Your loan is Approved!")
    #     time.sleep(5)
    #     with st.spinner('Predicting preferred Loan details ...'):
    #         time.sleep(5)
    #         result = Regressor()
    #         st.dataframe(result, use_container_width=st.session_state.use_container_width)
    #     st.balloons()

with st.sidebar:
    if st.button(f"APPLICATION STATUS {st.session_state.emoji}", on_click=random_emoji):
        st.header('Loan Application Status')
        with st.spinner('Analyzing the Provided Information ...'):
            time.sleep(5)
            result = Classifier()
            st.spinner(text="Analyzing the Information")

            if result == "Defaulter":
                st.write("Based on details provided, the user may default so loan is not approved, Thanks!")
                time.sleep(3)
                with st.spinner('Predicting preferred Loan details ...'):
                    time.sleep(5)
                    result=Regressor()
                    st.dataframe(result, use_container_width=st.session_state.use_container_width)
            st.balloons()

            if result == "Not Defaulter":
                st.write("Congratulations! Your loan is Approved!")
                time.sleep(5)
                with st.spinner('Predicting preferred Loan details ...'):
                    time.sleep(5)
                    result = Regressor()
                    st.dataframe(result, use_container_width=st.session_state.use_container_width)
            st.balloons()
    if st.button(f"CHECK YOUR FILLED DETAIL {st.session_state.emoji}", on_click=random_emoji):
        st.checkbox("Use container width", value=False, key="use_container_width")
        df = load_data()
        st.dataframe(df, use_container_width=st.session_state.use_container_width)
        st.balloons()
