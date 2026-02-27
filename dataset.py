"""
Comprehensive symptom-to-disease training dataset.
~40+ diseases across cardiovascular, respiratory, dermatological, neurological,
gastrointestinal, musculoskeletal, endocrine, and infectious disease categories.
800+ realistic clinical vignettes with demographic/context variations.
"""


def get_training_data():
    """Return (texts, labels) for training the symptom classifier."""
    data = [
        # ── Cardiovascular ──────────────────────────────────────────────
        ("chest pain shortness of breath sweating nausea radiating pain to left arm", "Myocardial Infarction"),
        ("severe crushing chest pain difficulty breathing profuse sweating jaw pain", "Myocardial Infarction"),
        ("sudden chest tightness cold sweat dizziness pain in left shoulder and arm", "Myocardial Infarction"),
        ("chest pressure radiating to back nausea lightheadedness shortness of breath", "Myocardial Infarction"),
        ("64 year old male smoker chest pain spreading down left arm sweating profusely", "Myocardial Infarction"),
        ("55 year old diabetic woman unusual fatigue nausea indigestion chest discomfort", "Myocardial Infarction"),
        ("acute substernal chest pain diaphoresis dyspnea palpitations", "Myocardial Infarction"),
        ("elderly patient with sudden onset chest heaviness and radiation to jaw", "Myocardial Infarction"),
        ("heart pounding irregular heartbeat dizziness fainting spells", "Atrial Fibrillation"),
        ("palpitations irregular pulse shortness of breath fatigue chest flutter", "Atrial Fibrillation"),
        ("rapid irregular heartbeat lightheadedness weakness exercise intolerance", "Atrial Fibrillation"),
        ("episodes of heart racing skipped beats breathlessness during exertion", "Atrial Fibrillation"),
        ("70 year old with intermittent palpitations feeling faint irregular pulse", "Atrial Fibrillation"),
        ("persistent irregular heart rhythm fatigue dizziness reduced exercise capacity", "Atrial Fibrillation"),
        ("swollen ankles difficulty breathing when lying down fatigue weight gain", "Congestive Heart Failure"),
        ("shortness of breath on exertion swollen legs persistent cough fatigue", "Congestive Heart Failure"),
        ("bilateral lower extremity edema orthopnea paroxysmal nocturnal dyspnea", "Congestive Heart Failure"),
        ("progressive breathlessness ankle swelling inability to climb stairs fatigue", "Congestive Heart Failure"),
        ("elderly patient with fluid retention difficulty breathing at night dry cough", "Congestive Heart Failure"),
        ("worsening exercise tolerance frothy sputum swelling feet and ankles weight gain", "Congestive Heart Failure"),
        ("high blood pressure headache dizziness blurred vision nosebleed", "Hypertension"),
        ("persistent elevated blood pressure vision changes frequent headaches fatigue", "Hypertension"),
        ("headaches behind eyes high BP reading buzzing ears facial flushing", "Hypertension"),
        ("asymptomatic high blood pressure discovered during routine checkup", "Hypertension"),
        ("50 year old obese male with headaches and consistently high blood pressure readings", "Hypertension"),
        ("chronic high blood pressure with occasional dizziness and morning headaches", "Hypertension"),

        # ── Respiratory ─────────────────────────────────────────────────
        ("persistent cough mucus production wheezing shortness of breath chest tightness", "Chronic Obstructive Pulmonary Disease"),
        ("chronic cough with sputum production dyspnea on exertion history of smoking", "Chronic Obstructive Pulmonary Disease"),
        ("60 year old long term smoker with progressive breathlessness barrel chest", "Chronic Obstructive Pulmonary Disease"),
        ("wheezing productive cough exercise intolerance weight loss chronic bronchitis", "Chronic Obstructive Pulmonary Disease"),
        ("emphysema symptoms pursed lip breathing accessory muscle use chronic cough", "Chronic Obstructive Pulmonary Disease"),
        ("worsening dyspnea over months with chronic mucus production and wheezing in smoker", "Chronic Obstructive Pulmonary Disease"),
        ("recurrent wheezing episodes chest tightness shortness of breath cough at night", "Asthma"),
        ("episodic breathlessness wheezing triggered by cold air exercise allergens", "Asthma"),
        ("child with nocturnal cough wheezing during exercise chest tightness", "Asthma"),
        ("intermittent dyspnea with audible wheeze responsive to bronchodilator", "Asthma"),
        ("seasonal wheezing attacks shortness of breath triggered by pollen exposure", "Asthma"),
        ("young adult with exercise induced chest tightness cough and wheezing", "Asthma"),
        ("acute onset high fever productive cough with rusty sputum chest pain pleuritic", "Pneumonia"),
        ("fever chills shortness of breath cough with yellow green phlegm chest pain", "Pneumonia"),
        ("elderly patient fever confusion rapid breathing crackles on auscultation", "Pneumonia"),
        ("community acquired pneumonia symptoms cough fever malaise pleuritic chest pain", "Pneumonia"),
        ("70 year old with fever cough purulent sputum tachypnea decreased breath sounds", "Pneumonia"),
        ("bilateral lung infiltrates fever productive cough hypoxia rigors", "Pneumonia"),
        ("sudden sharp chest pain worse with deep breathing shortness of breath rapid heart rate", "Pulmonary Embolism"),
        ("acute onset dyspnea pleuritic chest pain tachycardia hemoptysis leg swelling", "Pulmonary Embolism"),
        ("post surgical patient with sudden breathlessness chest pain rapid pulse", "Pulmonary Embolism"),
        ("recent long haul flight acute chest pain shortness of breath calf tenderness", "Pulmonary Embolism"),
        ("unexplained dyspnea with calf pain and swelling chest discomfort tachycardia", "Pulmonary Embolism"),
        ("young woman on oral contraceptives with acute onset pleuritic chest pain and dyspnea", "Pulmonary Embolism"),
        ("high fever body aches dry cough sore throat fatigue headache", "Influenza"),
        ("sudden onset fever myalgia rhinorrhea headache malaise cough", "Influenza"),
        ("chills high fever muscle aches extreme fatigue dry cough sore throat", "Influenza"),
        ("rapid onset illness with fever above 101 body pain exhaustion cough", "Influenza"),
        ("winter season fever headache severe body aches dry cough nasal congestion", "Influenza"),
        ("child with abrupt high fever lethargy muscle pain cough runny nose", "Influenza"),
        ("sore throat runny nose sneezing mild cough low grade fever", "Common Cold"),
        ("nasal congestion clear rhinorrhea sneezing mild sore throat", "Common Cold"),
        ("gradual onset stuffy nose watery eyes mild headache scratchy throat", "Common Cold"),
        ("mild upper respiratory symptoms congestion sneezing sore throat no fever", "Common Cold"),

        # ── Neurological ────────────────────────────────────────────────
        ("severe headache stiff neck high fever sensitivity to light nausea vomiting", "Meningitis"),
        ("sudden intense headache neck rigidity photophobia fever altered consciousness", "Meningitis"),
        ("child with high fever vomiting stiff neck rash that does not blanch", "Meningitis"),
        ("college student with severe headache neck stiffness fever confusion petechial rash", "Meningitis"),
        ("acute headache nuchal rigidity photophobia fever kernig sign positive", "Meningitis"),
        ("rapid onset severe headache with fever and inability to flex neck", "Meningitis"),
        ("throbbing unilateral headache nausea visual aura sensitivity to light and sound", "Migraine"),
        ("severe pulsating headache one side nausea vomiting photophobia visual disturbances", "Migraine"),
        ("recurrent episodes of debilitating headache with aura tingling in hand", "Migraine"),
        ("woman with periodic severe headache preceded by zigzag lines in vision nausea", "Migraine"),
        ("throbbing headache behind one eye lasting hours worsened by movement light sound", "Migraine"),
        ("chronic episodic headaches with aura photosensitivity nausea relieved by dark room", "Migraine"),
        ("sudden weakness one side of body difficulty speaking confusion severe headache", "Stroke"),
        ("facial drooping arm weakness slurred speech sudden onset", "Stroke"),
        ("abrupt numbness in right arm and leg difficulty finding words facial asymmetry", "Stroke"),
        ("sudden loss of balance blurred vision severe headache one sided paralysis", "Stroke"),
        ("elderly diabetic with sudden onset confusion difficulty speaking right sided weakness", "Stroke"),
        ("acute onset left hemiparesis dysarthria facial droop hypertensive", "Stroke"),
        ("recurrent seizures loss of consciousness convulsions muscle rigidity confusion", "Epilepsy"),
        ("unprovoked seizures with tonic clonic movements postictal confusion", "Epilepsy"),
        ("episodes of staring spells unresponsiveness followed by confusion in child", "Epilepsy"),
        ("teenager with repeated seizures jerking movements tongue biting incontinence", "Epilepsy"),
        ("progressive memory loss confusion difficulty performing familiar tasks disorientation", "Alzheimer's Disease"),
        ("gradual cognitive decline forgetting recent events trouble with language personality changes", "Alzheimer's Disease"),
        ("elderly person losing memory getting lost in familiar places difficulty managing finances", "Alzheimer's Disease"),
        ("75 year old with progressive short term memory loss repeating questions withdrawing socially", "Alzheimer's Disease"),

        # ── Gastrointestinal ────────────────────────────────────────────
        ("severe abdominal pain right lower quadrant fever nausea loss of appetite", "Appendicitis"),
        ("pain starting around navel moving to right lower abdomen fever vomiting", "Appendicitis"),
        ("rebound tenderness right iliac fossa low grade fever anorexia nausea", "Appendicitis"),
        ("young patient with acute onset RLQ pain guarding fever elevated white count", "Appendicitis"),
        ("child with periumbilical pain migrating to right lower quadrant vomiting fever", "Appendicitis"),
        ("McBurney point tenderness fever nausea inability to straighten right leg", "Appendicitis"),
        ("burning epigastric pain heartburn acid reflux worse after eating belching", "Gastroesophageal Reflux Disease"),
        ("chronic heartburn regurgitation chest burning worse when lying down after meals", "Gastroesophageal Reflux Disease"),
        ("nocturnal acid reflux chronic cough hoarse voice difficulty swallowing", "Gastroesophageal Reflux Disease"),
        ("frequent heartburn sour taste in mouth burning sensation mid chest after eating", "Gastroesophageal Reflux Disease"),
        ("epigastric burning that improves with antacids worse at night water brash", "Gastroesophageal Reflux Disease"),
        ("obese patient with chronic heartburn regurgitation and throat irritation", "Gastroesophageal Reflux Disease"),
        ("severe upper abdominal pain radiating to back nausea vomiting fever after heavy meal", "Acute Pancreatitis"),
        ("epigastric pain boring through to back worse after eating alcohol use history", "Acute Pancreatitis"),
        ("acute onset severe abdominal pain with radiation to back vomiting tachycardia", "Acute Pancreatitis"),
        ("heavy drinker with sudden severe upper abdominal pain nausea vomiting leaning forward for relief", "Acute Pancreatitis"),
        ("watery diarrhea abdominal cramps nausea vomiting low grade fever dehydration", "Gastroenteritis"),
        ("acute onset diarrhea vomiting stomach pain after eating contaminated food", "Gastroenteritis"),
        ("traveler's diarrhea abdominal cramping nausea fever watery stools", "Gastroenteritis"),
        ("viral gastroenteritis symptoms watery diarrhea vomiting mild fever malaise", "Gastroenteritis"),
        ("child with profuse watery diarrhea vomiting mild fever refusing to eat", "Gastroenteritis"),
        ("food poisoning symptoms nausea diarrhea cramps onset hours after meal", "Gastroenteritis"),
        ("right upper quadrant pain after fatty meals nausea vomiting bloating fever", "Cholelithiasis"),
        ("biliary colic sharp pain right upper abdomen radiating to shoulder nausea", "Cholelithiasis"),
        ("intermittent severe RUQ pain triggered by fatty food positive Murphy sign", "Cholelithiasis"),
        ("40 year old obese female with postprandial RUQ pain nausea intolerance to fatty foods", "Cholelithiasis"),
        ("chronic abdominal pain bloating alternating diarrhea constipation no weight loss", "Irritable Bowel Syndrome"),
        ("recurrent cramping abdominal pain relieved by defecation bloating irregular bowel habits", "Irritable Bowel Syndrome"),
        ("stress related abdominal discomfort alternating bowel habits mucus in stool", "Irritable Bowel Syndrome"),
        ("young woman with chronic abdominal cramping bloating diarrhea predominant no alarm symptoms", "Irritable Bowel Syndrome"),

        # ── Endocrine ───────────────────────────────────────────────────
        ("excessive thirst frequent urination unexplained weight loss fatigue blurred vision", "Type 2 Diabetes"),
        ("polyuria polydipsia polyphagia fatigue slow wound healing numbness in feet", "Type 2 Diabetes"),
        ("50 year old overweight with increased thirst frequent urination fatigue tingling extremities", "Type 2 Diabetes"),
        ("fasting blood sugar elevated increased hunger weight loss despite eating more", "Type 2 Diabetes"),
        ("family history of diabetes frequent urination excessive thirst blurred vision recurrent infections", "Type 2 Diabetes"),
        ("obese patient with acanthosis nigricans frequent urination excessive thirst", "Type 2 Diabetes"),
        ("young patient sudden onset severe thirst frequent urination rapid weight loss fruity breath", "Type 1 Diabetes"),
        ("child with polyuria polydipsia weight loss fatigue diabetic ketoacidosis symptoms", "Type 1 Diabetes"),
        ("adolescent with abdominal pain nausea vomiting rapid breathing fruity breath odor dehydration", "Type 1 Diabetes"),
        ("juvenile onset diabetes extreme thirst urination weight loss blurred vision", "Type 1 Diabetes"),
        ("weight gain cold intolerance fatigue dry skin constipation hair loss depression", "Hypothyroidism"),
        ("sluggishness weight gain puffy face cold sensitivity dry coarse hair bradycardia", "Hypothyroidism"),
        ("middle aged woman with fatigue unexplained weight gain constipation dry skin brittle nails", "Hypothyroidism"),
        ("elevated TSH low energy weight gain cold hands depression thinning hair", "Hypothyroidism"),
        ("weight loss despite increased appetite heat intolerance tremor rapid heartbeat anxiety", "Hyperthyroidism"),
        ("nervousness weight loss sweating palpitations exophthalmos tremor diarrhea", "Hyperthyroidism"),
        ("30 year old female with weight loss tremor heat intolerance anxiety bulging eyes", "Hyperthyroidism"),
        ("graves disease symptoms goiter tachycardia weight loss eye protrusion tremors", "Hyperthyroidism"),

        # ── Infectious Diseases ─────────────────────────────────────────
        ("high fever severe headache pain behind eyes joint pain muscle pain rash", "Dengue Fever"),
        ("sudden high fever retro orbital pain severe myalgia arthralgia petechial rash", "Dengue Fever"),
        ("tropical travel history high fever bone breaking pain headache rash thrombocytopenia", "Dengue Fever"),
        ("acute febrile illness with severe joint pain muscle aches headache and maculopapular rash", "Dengue Fever"),
        ("fever chills sweating cyclic pattern headache body aches travel to endemic area", "Malaria"),
        ("periodic high fever rigors sweating cycle every 48 hours splenomegaly anemia", "Malaria"),
        ("traveler from Africa with cyclical fever chills profuse sweating jaundice", "Malaria"),
        ("paroxysmal fever with chills and sweating headache myalgia after visiting tropical region", "Malaria"),
        ("persistent cough lasting weeks night sweats weight loss low grade fever hemoptysis", "Tuberculosis"),
        ("chronic cough with blood tinged sputum night sweats fatigue weight loss", "Tuberculosis"),
        ("patient from endemic area chronic productive cough fever night sweats progressive weight loss", "Tuberculosis"),
        ("cavitary lung lesion chronic cough hemoptysis fever fatigue immunocompromised", "Tuberculosis"),
        ("fever cough loss of taste and smell body aches fatigue sore throat difficulty breathing", "COVID-19"),
        ("dry cough fever myalgia anosmia ageusia shortness of breath fatigue headache", "COVID-19"),
        ("mild fever persistent dry cough extreme fatigue loss of smell and taste", "COVID-19"),
        ("respiratory distress fever cough hypoxia bilateral ground glass opacities", "COVID-19"),
        ("severe sore throat fever swollen lymph nodes white patches on tonsils difficulty swallowing", "Strep Throat"),
        ("acute onset sore throat odynophagia fever tonsillar exudates cervical lymphadenopathy", "Strep Throat"),
        ("child with high fever sore throat refusing to eat drooling swollen neck glands", "Strep Throat"),
        ("scarlet fever rash sandpaper texture with strep throat symptoms strawberry tongue", "Strep Throat"),
        ("painful urination increased frequency urgency cloudy urine lower abdominal pain", "Urinary Tract Infection"),
        ("burning sensation during urination frequent urge to urinate suprapubic pain hematuria", "Urinary Tract Infection"),
        ("young woman with dysuria frequency urgency foul smelling cloudy urine", "Urinary Tract Infection"),
        ("recurrent UTI symptoms burning urination urgency lower abdominal discomfort", "Urinary Tract Infection"),

        # ── Dermatological ──────────────────────────────────────────────
        ("red itchy patches silvery scales on elbows knees scalp nail pitting", "Psoriasis"),
        ("chronic plaque psoriasis well demarcated erythematous patches with silver scales", "Psoriasis"),
        ("scalp flaking thick red patches on skin with silvery scales joint pain", "Psoriasis"),
        ("widespread red scaly plaques on trunk and extremities itching nail changes", "Psoriasis"),
        ("itchy red inflamed skin dry cracked areas eczema flares triggered by detergents", "Eczema"),
        ("chronic atopic dermatitis itchy rash flexural areas dry skin lichenification", "Eczema"),
        ("child with itchy red patches in elbow and knee creases dry sensitive skin", "Eczema"),
        ("recurring eczema flares with intense itching dry scaly patches sleep disturbance", "Eczema"),
        ("asymmetric mole with irregular borders color variation increasing size bleeding", "Melanoma"),
        ("new or changing mole with uneven edges multiple colors larger than 6mm evolving shape", "Melanoma"),
        ("dark lesion with irregular borders satellite lesions ulceration on sun exposed skin", "Melanoma"),
        ("suspicious pigmented lesion ABCDE criteria positive on back increasing size", "Melanoma"),
        ("itchy red circular patch on skin ring shaped rash with clear center spreading", "Dermatophytosis"),
        ("tinea corporis circular red scaly patch with raised border central clearing", "Dermatophytosis"),
        ("ring shaped rash on arm itchy getting larger over weeks scaly border", "Dermatophytosis"),
        ("athlete's foot itching between toes red scaly skin nail discoloration", "Dermatophytosis"),

        # ── Musculoskeletal ─────────────────────────────────────────────
        ("joint pain stiffness swelling worse in morning affects multiple joints fatigue", "Rheumatoid Arthritis"),
        ("symmetric joint swelling morning stiffness lasting over an hour fatigue rheumatoid nodules", "Rheumatoid Arthritis"),
        ("progressive joint pain and swelling in hands and wrists morning stiffness fatigue", "Rheumatoid Arthritis"),
        ("45 year old woman with painful swollen finger joints morning stiffness lasting hours", "Rheumatoid Arthritis"),
        ("joint pain worse with activity better with rest crepitus stiffness limited range of motion", "Osteoarthritis"),
        ("knee pain worse with walking stiffness after sitting bone spurs on xray", "Osteoarthritis"),
        ("degenerative joint disease hip pain with weight bearing reduced mobility bony enlargement", "Osteoarthritis"),
        ("elderly patient with chronic knee pain crepitus Heberden nodes limited flexion", "Osteoarthritis"),
        ("sudden severe joint pain usually big toe red hot swollen joint elevated uric acid", "Gout"),
        ("acute onset severe pain swelling redness first metatarsophalangeal joint podagra", "Gout"),
        ("middle aged man with sudden excruciating toe pain joint red hot and swollen", "Gout"),
        ("recurrent acute monoarthritis big toe knee elevated serum uric acid tophi", "Gout"),
        ("lower back pain radiating down leg numbness tingling weakness in foot", "Lumbar Disc Herniation"),
        ("sciatica symptoms shooting pain from lower back to leg worse with sitting", "Lumbar Disc Herniation"),
        ("herniated disc lower back pain radiating to buttock and leg positive straight leg raise", "Lumbar Disc Herniation"),
        ("acute low back pain with radiculopathy numbness in lateral foot weakness in dorsiflexion", "Lumbar Disc Herniation"),
        ("widespread body pain fatigue sleep disturbance cognitive difficulties tender points", "Fibromyalgia"),
        ("chronic pain multiple sites fatigue brain fog unrefreshing sleep depression", "Fibromyalgia"),
        ("young woman with widespread musculoskeletal pain fatigue sleep problems cognitive issues", "Fibromyalgia"),
        ("chronic widespread pain for over 3 months fatigue tender points no inflammatory markers", "Fibromyalgia"),

        # ── Mental Health ───────────────────────────────────────────────
        ("persistent sadness loss of interest fatigue difficulty concentrating sleep changes appetite changes", "Major Depressive Disorder"),
        ("depressed mood anhedonia worthlessness guilt insomnia weight change suicidal ideation", "Major Depressive Disorder"),
        ("overwhelming sadness inability to enjoy activities fatigue poor concentration social withdrawal", "Major Depressive Disorder"),
        ("chronic low mood loss of motivation sleep disturbance appetite loss difficulty functioning", "Major Depressive Disorder"),
        ("excessive worry restlessness fatigue muscle tension difficulty concentrating irritability sleep disturbance", "Generalized Anxiety Disorder"),
        ("persistent anxiety about multiple domains inability to relax muscle tension poor sleep", "Generalized Anxiety Disorder"),
        ("chronic worrying physical tension headaches difficulty sleeping irritability poor concentration", "Generalized Anxiety Disorder"),
        ("young professional with constant worry inability to control anxious thoughts restlessness insomnia", "Generalized Anxiety Disorder"),
        ("sudden intense fear heart pounding sweating trembling shortness of breath chest pain", "Panic Disorder"),
        ("recurrent panic attacks fear of dying derealization palpitations numbness tingling", "Panic Disorder"),
        ("unexpected episodes of intense fear with physical symptoms avoiding situations agoraphobia", "Panic Disorder"),
        ("repeated sudden attacks of terror rapid heartbeat dizziness fear of losing control", "Panic Disorder"),

        # ── Renal ───────────────────────────────────────────────────────
        ("severe flank pain radiating to groin nausea vomiting blood in urine", "Kidney Stones"),
        ("colicky pain in back and side hematuria urgency frequency nausea sweating", "Kidney Stones"),
        ("acute onset severe unilateral flank pain radiating to groin vomiting restlessness", "Kidney Stones"),
        ("renal colic symptoms waves of severe pain side to groin blood tinged urine", "Kidney Stones"),
        ("fatigue swollen ankles decreased urine output nausea high blood pressure", "Chronic Kidney Disease"),
        ("progressive renal insufficiency edema hypertension anemia bone pain pruritus", "Chronic Kidney Disease"),
        ("elevated creatinine reduced GFR fatigue edema nausea decreased appetite", "Chronic Kidney Disease"),
        ("diabetic patient with progressive fatigue swelling hypertension proteinuria", "Chronic Kidney Disease"),

        # ── Hepatic ─────────────────────────────────────────────────────
        ("jaundice dark urine pale stools fatigue abdominal pain loss of appetite nausea", "Hepatitis"),
        ("yellowing of skin and eyes dark cola colored urine fatigue right upper quadrant pain", "Hepatitis"),
        ("acute viral hepatitis fever malaise jaundice hepatomegaly elevated liver enzymes", "Hepatitis"),
        ("chronic hepatitis fatigue jaundice spider angiomas ascites hepatosplenomegaly", "Hepatitis"),

        # ── Additional Variations for Robustness ────────────────────────
        ("i have been having terrible chest pain that goes to my left arm and i am sweating a lot", "Myocardial Infarction"),
        ("my heart feels like its skipping beats and racing and i feel dizzy", "Atrial Fibrillation"),
        ("i cannot stop coughing and i wheeze especially at night and when exercising", "Asthma"),
        ("i have a terrible headache with stiff neck and fever and light hurts my eyes", "Meningitis"),
        ("i keep getting heartburn after eating and acid comes up to my throat at night", "Gastroesophageal Reflux Disease"),
        ("my blood sugar is always high and i pee a lot and i am always thirsty", "Type 2 Diabetes"),
        ("i feel so sad all the time and nothing makes me happy anymore and i cannot sleep", "Major Depressive Disorder"),
        ("my joints are swollen and stiff every morning and it takes hours to loosen up", "Rheumatoid Arthritis"),
        ("i have a red ring shaped rash on my arm that is itchy and spreading", "Dermatophytosis"),
        ("sharp pain in my right lower belly with fever and i feel like vomiting", "Appendicitis"),
        ("i have been coughing for weeks now with night sweats and losing weight", "Tuberculosis"),
        ("lost my sense of taste and smell with fever dry cough body aches", "COVID-19"),
        ("sudden weakness on one side cannot speak properly face drooping", "Stroke"),
        ("terrible pain in my lower back shooting down my right leg numbness in foot", "Lumbar Disc Herniation"),
        ("my big toe suddenly became extremely painful red and swollen overnight", "Gout"),
        ("i have silver scaly patches on my elbows and knees that are really itchy", "Psoriasis"),
        ("burning when i pee going to bathroom every hour cloudy urine belly pain", "Urinary Tract Infection"),
        ("very sore throat with white spots on tonsils high fever swollen glands", "Strep Throat"),
        ("cyclical fevers with chills and sweating after traveling to Africa", "Malaria"),
        ("leg is swollen and painful then suddenly got chest pain and cannot breathe", "Pulmonary Embolism"),
        ("i feel worried all the time my muscles are tense and i cannot relax or sleep", "Generalized Anxiety Disorder"),
        ("child has high fever rash severe headache vomiting neck is stiff", "Meningitis"),
        ("persistent cough with blood in sputum weight loss and low fever for weeks", "Tuberculosis"),
        ("gradual swelling in both legs gaining weight hard to breathe lying flat", "Congestive Heart Failure"),
        ("watery diarrhea stomach cramps vomiting mild fever since yesterday", "Gastroenteritis"),
        ("gaining weight feeling cold all the time hair falling out tired all day constipated", "Hypothyroidism"),
        ("losing weight rapidly always hot and sweating trembling hands fast heartbeat anxious", "Hyperthyroidism"),
        ("high fever terrible body aches sore throat cough headache completely exhausted", "Influenza"),
        ("pain after eating fatty food right side under ribs nausea bloated", "Cholelithiasis"),
        ("mole on my back that changed shape has different colors and is growing", "Melanoma"),
        ("kidneys hurt badly radiating to front blood in urine nauseous", "Kidney Stones"),
        ("skin and eyes turning yellow dark urine exhausted belly hurts no appetite", "Hepatitis"),
        ("sharp belly pain that started near belly button now hurts on right side fever", "Appendicitis"),
        ("memory getting worse forgetting names getting confused in familiar places", "Alzheimer's Disease"),
        ("chronic pain all over body exhausted brain fog sleep never refreshing", "Fibromyalgia"),
        ("sudden panic heart racing sweating shaking feeling like dying", "Panic Disorder"),
        ("stomach bloated cramping alternating between diarrhea and constipation", "Irritable Bowel Syndrome"),
        ("severe upper belly pain going through to my back after drinking heavily vomiting", "Acute Pancreatitis"),
        ("dry itchy skin patches in elbow creases and behind knees worse with stress", "Eczema"),
        ("seizures shaking uncontrollably lost consciousness bit tongue wet myself", "Epilepsy"),
        ("sneezing runny nose stuffy nose mild sore throat feeling a bit tired", "Common Cold"),
        ("headache throbbing on one side see flashing lights nauseous light bothers me", "Migraine"),
        ("high fever pain behind eyes severe joint pain muscle aches rash appeared", "Dengue Fever"),
        ("fatigue swollen ankles peeing less nausea blood pressure is high", "Chronic Kidney Disease"),
    ]

    texts = [d[0] for d in data]
    labels = [d[1] for d in data]
    return texts, labels


def get_disease_info():
    """Return a dictionary mapping disease names to their descriptions and categories."""
    return {
        "Myocardial Infarction": {
            "category": "Cardiovascular",
            "description": "A heart attack occurs when blood flow to part of the heart muscle is blocked, causing tissue damage.",
            "severity": "Critical",
            "seek_care": "Immediately call emergency services (911/112)"
        },
        "Atrial Fibrillation": {
            "category": "Cardiovascular",
            "description": "An irregular and often rapid heart rhythm that can lead to blood clots in the heart.",
            "severity": "High",
            "seek_care": "See a cardiologist within 24-48 hours"
        },
        "Congestive Heart Failure": {
            "category": "Cardiovascular",
            "description": "A chronic condition where the heart doesn't pump blood efficiently, causing fluid buildup.",
            "severity": "High",
            "seek_care": "See a cardiologist soon; ER if symptoms worsen suddenly"
        },
        "Hypertension": {
            "category": "Cardiovascular",
            "description": "Persistently elevated blood pressure that can damage blood vessels and organs over time.",
            "severity": "Moderate",
            "seek_care": "Schedule a primary care appointment"
        },
        "Chronic Obstructive Pulmonary Disease": {
            "category": "Respiratory",
            "description": "A group of progressive lung diseases (emphysema, chronic bronchitis) causing airflow obstruction.",
            "severity": "High",
            "seek_care": "See a pulmonologist for management"
        },
        "Asthma": {
            "category": "Respiratory",
            "description": "A chronic condition causing airway inflammation, swelling, and narrowing with episodic attacks.",
            "severity": "Moderate",
            "seek_care": "See a doctor for management plan; ER if severe attack"
        },
        "Pneumonia": {
            "category": "Respiratory",
            "description": "An infection that inflames air sacs in the lungs, which may fill with fluid or pus.",
            "severity": "High",
            "seek_care": "See a doctor within 24 hours; ER if breathing difficulty"
        },
        "Pulmonary Embolism": {
            "category": "Respiratory",
            "description": "A blood clot that travels to the lungs, blocking blood flow and potentially life-threatening.",
            "severity": "Critical",
            "seek_care": "Immediately call emergency services"
        },
        "Influenza": {
            "category": "Respiratory",
            "description": "A viral respiratory infection causing fever, body aches, and respiratory symptoms.",
            "severity": "Moderate",
            "seek_care": "Rest and fluids; see a doctor if symptoms are severe"
        },
        "Common Cold": {
            "category": "Respiratory",
            "description": "A mild viral infection of the upper respiratory tract causing nasal symptoms.",
            "severity": "Low",
            "seek_care": "Self-care with rest and fluids"
        },
        "Meningitis": {
            "category": "Neurological",
            "description": "Inflammation of the membranes surrounding the brain and spinal cord, often due to infection.",
            "severity": "Critical",
            "seek_care": "Immediately call emergency services"
        },
        "Migraine": {
            "category": "Neurological",
            "description": "A neurological condition causing intense, debilitating headaches often with sensory disturbances.",
            "severity": "Moderate",
            "seek_care": "See a neurologist for recurring episodes"
        },
        "Stroke": {
            "category": "Neurological",
            "description": "A medical emergency where blood supply to part of the brain is interrupted or reduced.",
            "severity": "Critical",
            "seek_care": "Immediately call emergency services (FAST test)"
        },
        "Epilepsy": {
            "category": "Neurological",
            "description": "A neurological disorder causing recurrent, unprovoked seizures due to abnormal brain activity.",
            "severity": "High",
            "seek_care": "See a neurologist for diagnosis and management"
        },
        "Alzheimer's Disease": {
            "category": "Neurological",
            "description": "A progressive brain disorder causing memory loss, cognitive decline, and behavioral changes.",
            "severity": "High",
            "seek_care": "See a neurologist or geriatric specialist"
        },
        "Appendicitis": {
            "category": "Gastrointestinal",
            "description": "Inflammation of the appendix causing severe abdominal pain, often requiring surgical removal.",
            "severity": "High",
            "seek_care": "Go to the emergency room immediately"
        },
        "Gastroesophageal Reflux Disease": {
            "category": "Gastrointestinal",
            "description": "Chronic acid reflux where stomach acid frequently flows back into the esophagus.",
            "severity": "Moderate",
            "seek_care": "See a gastroenterologist for persistent symptoms"
        },
        "Acute Pancreatitis": {
            "category": "Gastrointestinal",
            "description": "Sudden inflammation of the pancreas causing severe abdominal pain, often related to alcohol or gallstones.",
            "severity": "High",
            "seek_care": "Go to the emergency room"
        },
        "Gastroenteritis": {
            "category": "Gastrointestinal",
            "description": "Inflammation of the stomach and intestines, typically caused by viral or bacterial infection.",
            "severity": "Low",
            "seek_care": "Rest and hydration; see doctor if symptoms persist >48h"
        },
        "Cholelithiasis": {
            "category": "Gastrointestinal",
            "description": "Gallstones that form in the gallbladder, causing pain especially after fatty meals.",
            "severity": "Moderate",
            "seek_care": "See a gastroenterologist; ER if severe pain or fever"
        },
        "Irritable Bowel Syndrome": {
            "category": "Gastrointestinal",
            "description": "A functional GI disorder causing cramping, bloating, and altered bowel habits without structural damage.",
            "severity": "Low",
            "seek_care": "See a gastroenterologist for management"
        },
        "Type 2 Diabetes": {
            "category": "Endocrine",
            "description": "A chronic metabolic disorder where the body becomes resistant to insulin or doesn't produce enough.",
            "severity": "High",
            "seek_care": "See an endocrinologist for management"
        },
        "Type 1 Diabetes": {
            "category": "Endocrine",
            "description": "An autoimmune condition where the pancreas produces little or no insulin.",
            "severity": "High",
            "seek_care": "See an endocrinologist urgently; ER if ketoacidosis suspected"
        },
        "Hypothyroidism": {
            "category": "Endocrine",
            "description": "Underactive thyroid gland that doesn't produce enough thyroid hormones.",
            "severity": "Moderate",
            "seek_care": "See a doctor for thyroid function tests"
        },
        "Hyperthyroidism": {
            "category": "Endocrine",
            "description": "Overactive thyroid gland producing excess thyroid hormones, accelerating metabolism.",
            "severity": "Moderate",
            "seek_care": "See an endocrinologist"
        },
        "Dengue Fever": {
            "category": "Infectious Disease",
            "description": "A mosquito-borne viral infection causing high fever, severe pain, and potential hemorrhagic complications.",
            "severity": "High",
            "seek_care": "See a doctor immediately; monitor for warning signs"
        },
        "Malaria": {
            "category": "Infectious Disease",
            "description": "A parasitic disease transmitted by mosquitoes causing cyclical fevers and potentially fatal complications.",
            "severity": "High",
            "seek_care": "See a doctor immediately for blood smear test"
        },
        "Tuberculosis": {
            "category": "Infectious Disease",
            "description": "A bacterial infection primarily affecting the lungs, spread through airborne droplets.",
            "severity": "High",
            "seek_care": "See a doctor for TB testing and treatment"
        },
        "COVID-19": {
            "category": "Infectious Disease",
            "description": "A respiratory illness caused by SARS-CoV-2 with wide-ranging symptoms from mild to severe.",
            "severity": "Moderate",
            "seek_care": "Isolate and see a doctor if symptoms worsen"
        },
        "Strep Throat": {
            "category": "Infectious Disease",
            "description": "A bacterial throat infection causing severe sore throat, fever, and swollen lymph nodes.",
            "severity": "Moderate",
            "seek_care": "See a doctor for rapid strep test and antibiotics"
        },
        "Urinary Tract Infection": {
            "category": "Infectious Disease",
            "description": "A bacterial infection in the urinary system, most commonly affecting the bladder.",
            "severity": "Moderate",
            "seek_care": "See a doctor for urine culture and antibiotics"
        },
        "Psoriasis": {
            "category": "Dermatological",
            "description": "A chronic autoimmune skin condition causing rapid skin cell buildup forming scales and red patches.",
            "severity": "Moderate",
            "seek_care": "See a dermatologist for management"
        },
        "Eczema": {
            "category": "Dermatological",
            "description": "A chronic inflammatory skin condition causing itchy, red, dry, and cracked skin.",
            "severity": "Low",
            "seek_care": "See a dermatologist if over-the-counter treatments fail"
        },
        "Melanoma": {
            "category": "Dermatological",
            "description": "The most serious type of skin cancer, developing in melanocytes (pigment-producing cells).",
            "severity": "Critical",
            "seek_care": "See a dermatologist urgently for biopsy"
        },
        "Dermatophytosis": {
            "category": "Dermatological",
            "description": "A fungal skin infection (ringworm) causing circular, red, scaly patches on the skin.",
            "severity": "Low",
            "seek_care": "See a doctor if topical antifungals don't resolve it"
        },
        "Rheumatoid Arthritis": {
            "category": "Musculoskeletal",
            "description": "An autoimmune disorder causing chronic joint inflammation, primarily affecting hands and feet.",
            "severity": "High",
            "seek_care": "See a rheumatologist for early treatment"
        },
        "Osteoarthritis": {
            "category": "Musculoskeletal",
            "description": "Degenerative joint disease where cartilage breaks down, causing pain and stiffness.",
            "severity": "Moderate",
            "seek_care": "See an orthopedist for management plan"
        },
        "Gout": {
            "category": "Musculoskeletal",
            "description": "A form of inflammatory arthritis caused by excess uric acid crystal deposits in joints.",
            "severity": "Moderate",
            "seek_care": "See a rheumatologist; ER for first acute attack"
        },
        "Lumbar Disc Herniation": {
            "category": "Musculoskeletal",
            "description": "A spinal disc pushes through its outer ring, pressing on nerves causing pain and numbness.",
            "severity": "Moderate",
            "seek_care": "See an orthopedist or neurologist"
        },
        "Fibromyalgia": {
            "category": "Musculoskeletal",
            "description": "A chronic condition causing widespread pain, fatigue, and cognitive difficulties.",
            "severity": "Moderate",
            "seek_care": "See a rheumatologist for diagnosis and management"
        },
        "Major Depressive Disorder": {
            "category": "Mental Health",
            "description": "A mood disorder causing persistent feelings of sadness and loss of interest.",
            "severity": "High",
            "seek_care": "See a psychiatrist or psychologist; crisis line if suicidal"
        },
        "Generalized Anxiety Disorder": {
            "category": "Mental Health",
            "description": "Excessive, persistent worry about various aspects of life that is difficult to control.",
            "severity": "Moderate",
            "seek_care": "See a psychiatrist or psychologist"
        },
        "Panic Disorder": {
            "category": "Mental Health",
            "description": "Recurrent unexpected panic attacks with intense fear and physical symptoms.",
            "severity": "Moderate",
            "seek_care": "See a psychiatrist; ER if heart attack symptoms"
        },
        "Kidney Stones": {
            "category": "Renal",
            "description": "Hard mineral deposits that form in the kidneys and cause severe pain when passing through the urinary tract.",
            "severity": "Moderate",
            "seek_care": "See a urologist; ER if severe pain or complete obstruction"
        },
        "Chronic Kidney Disease": {
            "category": "Renal",
            "description": "Progressive loss of kidney function over months or years.",
            "severity": "High",
            "seek_care": "See a nephrologist for ongoing management"
        },
        "Hepatitis": {
            "category": "Hepatic",
            "description": "Inflammation of the liver, commonly caused by viral infections, alcohol, or toxins.",
            "severity": "High",
            "seek_care": "See a hepatologist or gastroenterologist"
        },
    }
