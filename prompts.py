classification_prompt = """ Classify the text as labels: DRUG_INFO, HARMFUL, OTHER
    ======================================
    LABEL_MAPPER =
        "DRUG_INFO": [drug info, drug name],
        "HARMFUL_CONTENT": [hate speech, cuss words],
        "OTHER": ["out of scope","call me tomorrow", "call when you find out", "call me", "any"]
        ======================================
            Examples
            text: What is Levofloxacin
            labels: DRUG_INFO
            text: Difference between paracetamol and amox
            labels: DRUG_INFO
            text: Dropping my kids to school
            labels: OTHER
            text: no i actually would like to purchase it as soon as possible
            labels: OTHER
            text: no i actually would like to purchase it as soon as possible
            labels: OTHER
            text: 4 or 5
            labels: OTHER
            text: tomorrow morning 10 am
            labels: OTHER
            text: '{text}'
            labels:
            """



generation_prompt = (
    "Provide a humane answer for the '{inquiry_message}' from a customer who is looking for drug info. \n\n"
    "Use the provided feature information: '{context}' \n\n "
    "Avoid repeating phrases. \n\n"
)
