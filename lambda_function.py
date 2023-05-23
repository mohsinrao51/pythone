import stanza
import spacy_stanza
import negspacy
import pandas as pd 
print('before')
# from negspacy.negation import Negex
print('after')


def lambda_handler(event, _context):
    print('start lambda_handler')
    nlp_stanza = spacy_stanza.load_pipeline('en', package='mimic', processors={'ner':'i2b2'})
    nlp_stanza.add_pipe("negex", config={'ent_types': ['PROBLEM', 'TEST', 'TREATMENT']})
    #text = event['text']
    text = "--- page 4 --- 6/1/2015 1:39 PM Weight: 160 lb Height: 69.5 in Weight was reported by patient. Height was reported by patient. Body Surface Area: 1.89 m² Body Mass Index: 23.29 kg/m² Resp.: 18 (Unlabored) BP: 122/69 (Sitting, Left Arm, Standard) Physical Exam (Courtney Shands, MD; 6/1/2015 1:22 AM) The physical exam findings are as follows: General General Appearance - Not in acute distress. Orientation - Oriented to person, place and time. Build & Nutrition - Well nourished and Well developed. Head and Neck Head Head Shape - normocephalic, atraumatic. Chest and Lung Exam Chest and lung exam reveals - quiet, even and easy respiratory effort with no use of accessory muscles. Abdomen Inspection: Inspection of the abdomen reveals - No umbilical, inguinal, or incisional hernias. Contour - Normal. Incisional scars - No incisional scars. Palpation/Percussion Palpation and Percussion of the abdomen reveal - No costovertebral angle tenderness, Non Tender, No hepatosplenomegaly and No Palpable abdominal masses. Bladder - Non-palpable. Male Genitourinary Urethra: Characteristics - normal external urethral meatus at tip of penis. Discharge - None. Glans - Normal. Penis - Penile shaft is without rashes, lesions, palpable plaques. Scrotum - without lesions or rashes. Testes - Bilateral - Normal descended palpable testes, no testicluar mass, no hydrocele, no focal tendemess. Epididymis - Bilateral - Normal. Rectal Anorectal Exam: External - normal external exam. Internal - normal sphincter tone. Note: anterior rectal wall indurated Prostate - symmetric, no nodules, nontender. Prostate size - 20. Seminal Vesicle - Non Palpable. Assessment & Plan (Diana Landers; 6/1/2015 1:35 PM) FAMILY HX PROSTATE CA (V16.42) Impression: brother PROSTATE CANCER (185) Impression: T1c gleas 6, XRT 11/02 Current Plans PSA (PROSTATE SPECIFIC ANTIGEN) (84153) Schedule: Office visit/PSA 1 YR INFLUENZA IMMUNIZATION ADMINISTERED OR PREVIOUSLY RECEIVED (G8482) PNEUMOCOCCAL CONJUGATE VACCINE (4040F) TOBACCO NON-USER (1036F) LIST OF CURRENT MEDICATIONS (INCLUDES PRESCRIPTION, OVER-THE-COUNTER, HERBALS, VITAMIN/MINERAL/DIETARY INUTRITIONAL SUPPLEMENTS) DOCUMENTED BY THE PROVIDER, INQLUDING DRUG NAME. DOSAGE, FREQUENCY AND ROUTE (G8427) Pt Education - How to access health information online: discussed with patient and provided information. Medical Decision Making (Courtney Shands, MD; 6/1/2015 2:04 PM) Amount/complexity of data to be reviewed: - Order and/or review of lab test(s) Immunization Record (Diana Landers; 6/1/2015 1:38 PM) Immunization Type Immunization Order Date Funding Comments Influenza (3 years and up) Influenza (3 years and up) 2014 Pneumococcal (2 years and up) Pneumococcal (2 years and up) No Date Given 08/05/2015 10:31 am Robert D. Levine DOB 06/02/1940 Page 4/144"
    doc = nlp_stanza(event['text'])
    print('nlp done')

    df_summary = pd.DataFrame({'text': [], 'label': [], 'start': [], 'end': [], 'txt original': [],
                               'original sentence': []})

    print(df_summary)
    for ent in doc.ents:
        df_summary.loc[len(df_summary)] = [ent.text, ent.label_, ent.start_char, ent.end_char,
                                           text[int(ent.start_char): int(ent.end_char)], ent.sent]
    print(df_summary.to_string())
    return df_summary.to_string()
