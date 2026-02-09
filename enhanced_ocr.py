import os
import cv2
import numpy as np
import json
import re
from PIL import Image
from paddleocr import PaddleOCR, PPStructure
import torch
import difflib
import spacy
from fuzzywuzzy import fuzz, process
from skimage import exposure, filters
from skimage.filters import unsharp_mask
from skimage.morphology import disk

try:
    # Try to load spacy model for medical NER
    nlp = spacy.load("en_core_sci_md")
except:
    try:
        # Fall back to standard English model
        nlp = spacy.load("en_core_web_sm")
    except:
        nlp = None
        print("Warning: Spacy model not loaded. Using fallback text processing.")

# Initialize PaddleOCR with optimized parameters
ocr_engine = PaddleOCR(
    use_angle_cls=True,       # Enable text orientation detection
    lang='en',                # English language
    rec_algorithm='SVTR_LCNet',  # Use SVTR_LCNet for better handwritten text recognition
    rec_batch_num=6,          # Increased batch size for recognition
    det_limit_side_len=960,   # Limit size for detection to improve accuracy
    det_db_thresh=0.3,        # Lower threshold for detection to catch faint handwriting
    det_db_box_thresh=0.5,    # Adjusted box threshold
    max_text_length=100,       # Longer text length for prescriptions
    drop_score=0.5,           # Minimum confidence score
    use_gpu=torch.cuda.is_available(),  # Use GPU if available
    show_log=False            # Don't show log for production
)

# Medical dictionary for common prescription terms
MEDICATION_DICT = {
    # Common medications - Updated for Indian market
    "amoxicillin": ["amox", "amoxil", "amoxicil", "amoxicilin", "mox", "novamox", "almoxi", "wymox"],
    "paracetamol": ["paracet", "parcetamol", "acetaminophen", "tylenol", "crocin", "panadol", "dolo", "metacin", "calpol", "sumo", "febrex", "acepar", "pacimol"],
    "ibuprofen": ["ibuprofin", "ibu", "ibuprofen", "advil", "motrin", "nurofen", "brufen", "ibugesic", "combiflam"],
    "aspirin": ["asa", "acetylsalicylic", "aspr", "disprin", "ecotrin", "bayer", "loprin", "delisprin", "colsprin"],
    "lisinopril": ["lisin", "prinivil", "zestril", "qbrelis", "listril", "hipril", "zestopril"],
    "metformin": ["metform", "glucophage", "fortamet", "glumetza", "riomet", "glycomet", "obimet", "gluformin", "glyciphage"],
    "atorvastatin": ["lipitor", "atorva", "atorvastat", "lipibec", "atorlip", "atocor", "storvas"],
    "levothyroxine": ["synthroid", "levothy", "levothyrox", "levoxyl", "tirosint", "euthyrox", "thyronorm", "eltroxin"],
    "omeprazole": ["prilosec", "omepraz", "losec", "zegerid", "priosec", "omez", "ocid", "prazole"],
    "amlodipine": ["norvasc", "amlo", "amlod", "katerzia", "norvasc", "amlopress", "amlopres", "amlokind"],
    "metoprolol": ["lopressor", "toprol", "metopro", "toprol-xl", "betaloc", "metolar", "starpress"],
    "sertraline": ["zoloft", "sert", "sertra", "lustral", "serta", "daxid", "serlin"],
    "gabapentin": ["neurontin", "gaba", "gabap", "gralise", "horizant", "gabapin", "gaban", "progaba"],
    "hydrochlorothiazide": ["hctz", "hydrochlor", "microzide", "hydrodiuril", "hydrazide", "aquazide"],
    "simvastatin": ["zocor", "simvast", "simlup", "simcard", "simvotin", "zosta", "simgal"],
    "losartan": ["cozaar", "losart", "lavestra", "repace", "losar", "zaart", "covance"],
    "albuterol": ["proventil", "ventolin", "proair", "salbutamol", "asthalin", "ventofort", "aeromist"],
    "fluoxetine": ["prozac", "sarafem", "rapiflux", "prodep", "fludac", "flunil", "flunat"],
    "citalopram": ["celexa", "cipramil", "citalo", "celepram", "citalex", "citopam"],
    "pantoprazole": ["protonix", "pantoloc", "pantocid", "pantodac", "zipant", "pan"],
    "furosemide": ["lasix", "furos", "frusemide", "frusol", "lasix", "frusenex", "diucontin"],
    "rosuvastatin": ["crestor", "rosuvast", "rosuvas", "rovista", "rostor", "colver"],
    "escitalopram": ["lexapro", "cipralex", "nexito", "feliz", "stalopam", "nexito-forte"],
    "montelukast": ["singulair", "montek", "montair", "monticope", "romilast", "monty-lc"],
    "prednisone": ["deltasone", "predni", "orasone", "predcip", "omnacortil", "wysolone"],
    "warfarin": ["coumadin", "jantoven", "warf", "warfex", "warfrant", "uniwarfin"],
    "tramadol": ["ultram", "tram", "tramahexal", "tramazac", "domadol", "tramacip"],
    "azithromycin": ["zithromax", "azithro", "z-pak", "azith", "azee", "aziwok", "azimax", "zithrocin"],
    "ciprofloxacin": ["cipro", "ciloxan", "ciproxin", "ciplox", "ciprobid", "cifran", "ciprinol"],
    "lamotrigine": ["lamictal", "lamot", "lamotrigin", "lamogard", "lamitor", "lametec"],
    "venlafaxine": ["effexor", "venlaf", "venlor", "veniz", "ventab", "venlift"],
    "insulin": ["lantus", "humulin", "novolin", "humalog", "novolog", "tresiba", "insugen", "wosulin", "basalog", "insuman", "apidra"],
    "metronidazole": ["flagyl", "metro", "metrogel", "metrogyl", "metrozole", "aristogyl"],
    "naproxen": ["aleve", "naprosyn", "anaprox", "xenar", "napra", "napxen"],
    "doxycycline": ["vibramycin", "oracea", "doxy", "doxin", "biodoxi", "doxt"],
    "cetirizine": ["zyrtec", "cetryn", "cetriz", "alerid", "cetcip", "zirtin", "cetzine"],
    "diazepam": ["valium", "valpam", "dizac", "calmpose", "zepose", "sedopam"],
    "alprazolam": ["xanax", "alprax", "tafil", "alp", "alzolam", "zolax", "restyl", "trika"],
    "clonazepam": ["klonopin", "rivotril", "clon", "petril", "clonopam", "lonazep"],
    "carvedilol": ["coreg", "carvedil", "cardivas", "carca", "carvil", "carloc"],
    "fexofenadine": ["allegra", "telfast", "fexofine", "fexova", "agimfast", "allerfast"],
    "ranitidine": ["zantac", "ranit", "rantec", "aciloc", "zinetac", "histac"],
    "diclofenac": ["voltaren", "diclof", "diclomax", "voveran", "diclonac", "reactin"],
    "ceftriaxone": ["rocephin", "ceftri", "cefaxone", "inocef", "trixone", "monotax"],
    "cefixime": ["suprax", "cefi", "taxim", "unice", "cefispan", "omnicef"],
    "esomeprazole": ["nexium", "esotrex", "esopral", "nexpro", "raciper", "sompraz"],
    "clopidogrel": ["plavix", "clopid", "plagerine", "clopilet", "deplatt", "noklot"],
    
    # Adding more Indian medications
    "levocetirizine": ["xyzal", "levocet", "teczine", "levazeo", "xyzra", "uvnil"],
    "febuxostat": ["uloric", "febugat", "febuget", "zylobact", "febustat"],
    "telmisartan": ["micardis", "telma", "telsar", "sartel", "telvas", "cresar"],
    "folic acid": ["folate", "folvite", "folet", "folacin", "folitab", "obifolic"],
    "olmesartan": ["benicar", "olmat", "olmy", "benitec", "olmezest", "olsar"],
    "vildagliptin": ["galvus", "zomelis", "jalra", "vysov", "vildalip", "viladay"],
    "sitagliptin": ["januvia", "sitagen", "istamet", "janumet", "sitaglip", "trevia"],
    "metoprolol": ["lopresor", "metolar", "metocard", "betaloc", "meto-er", "metopro"],
    "glimepiride": ["amaryl", "glimpid", "glymex", "zoryl", "glimer", "diaglip"],
    "gliclazide": ["diamicron", "lycazid", "glizid", "reclide", "odinase", "glynase"],
    "ramipril": ["altace", "cardiopril", "cardace", "ramiril", "celapres", "ramace"],
    "nebivolol": ["bystolic", "nebistar", "nebicard", "nebilong", "nebilet", "nubeta"],
    "cilnidipine": ["cilacar", "cinod", "ciladay", "neudipine", "cilaheart", "cidip"],
    "rabeprazole": ["aciphex", "rablet", "rabicip", "razo", "raboz", "pepcia"],
    "dexamethasone": ["decadron", "dexona", "dexamycin", "dexacort", "decilone", "dexasone"],
    "doxofylline": ["doxolin", "synasma", "doxobid", "doxovent", "doxoril", "doxfree"],
    "deflazacort": ["dezacor", "defcort", "defza", "xenocort", "flacort", "defolet"],
    "ondansetron": ["zofran", "ondem", "emeset", "vomitrol", "zondem", "ondemet"],
    "domperidone": ["motilium", "domstal", "vomistop", "dompan", "dompy", "domcolic"],
    "pantoprazole": ["pantocid", "pantop", "pantodac", "panto", "pan-d", "pantozol"],
    "mefenamic acid": ["ponstan", "meftal", "meflam", "mefkind", "rafen", "mefgesic"],
    "aceclofenac": ["aceclo", "hifenac", "zerodol", "movon", "aceclo-plus", "acebid"],
    "nimesulide": ["nise", "nimulid", "nimek", "nimica", "nimcet", "nimsaid"],
    "hydroxyzine": ["atarax", "anxnil", "hyzox", "hydryllin", "anxipar", "hydrax"],
    "amlodipine + atenolol": ["amlokind-at", "amtas-at", "stamlo-beta", "tenolam", "amlopress-at"],
    "telmisartan + hydrochlorothiazide": ["telma-h", "telsar-h", "telvas-h", "tazloc-h", "telista-h"],
    "sulfamethoxazole + trimethoprim": ["bactrim", "septran", "cotrim", "sepmax", "oriprim"],
    "amoxicillin + clavulanic acid": ["augmentin", "moxclav", "megaclox", "clavam", "hiclav", "clavum"],
    "ofloxacin": ["oflox", "oflin", "tarivid", "zenflox", "oflacin", "exocin"],
    "torsemide": ["demadex", "dytor", "tide", "torlactone", "presage", "tomide"],
    "chlorthalidone": ["thalitone", "clorpres", "cloress", "natrilix", "thaloride"],
    "ivermectin": ["stromectol", "ivermect", "ivecop", "ivepred", "scabo", "ivernex"],
    "rifaximin": ["xifaxan", "rifagut", "rcifax", "rifakem", "rifamide"],
    "nitrofurantoin": ["furadantin", "niftran", "nitrofur", "furadoine", "nidantin"],
    "betahistine": ["serc", "vertin", "betaserc", "vertigo", "beta", "histiwel"],
    "etizolam": ["etilaam", "etizola", "sedekopan", "etizaa", "etzee", "etova"],
    "clotrimazole": ["candid", "clotri", "mycomax", "candiderma", "candifun", "clotop"],
    "ketoconazole": ["nizoral", "sebizole", "ketoz", "fungicide", "ketomac", "ketostar"],
    "fluconazole": ["diflucan", "flucz", "forcan", "syscan", "zocon", "flucos"],
    "pregabalin": ["lyrica", "pregeb", "maxgalin", "nervalin", "pregastar", "pregica"],
    "methylprednisolone": ["medrol", "methylpred", "depo-medrol", "solu-medrol", "depopred", "medrate"],
    "levetiracetam": ["keppra", "levesam", "levroxa", "levipil", "levecetam", "epictal"],
    
    # Common dosage units
    "milligram": ["mg", "mgs", "millig", "milligram"],
    "microgram": ["mcg", "µg", "microg"],
    "gram": ["g", "gm", "gms", "gram"],
    "milliliter": ["ml", "mls", "millil"],
    
    # Common frequency terms
    "once daily": ["qd", "od", "daily", "once a day", "1 time a day", "1x day"],
    "twice daily": ["bid", "bd", "twice a day", "2 times a day", "2x day"],
    "three times daily": ["tid", "tds", "3 times a day", "3x day"],
    "four times daily": ["qid", "qds", "4 times a day", "4x day"],
    "every morning": ["qam", "morn", "morning"],
    "every night": ["qhs", "qpm", "noct", "night", "bedtime", "bed time"],
    "every hour": ["q1h", "hourly"],
    "every 4 hours": ["q4h", "4 hourly", "every 4 hrs"],
    "every 6 hours": ["q6h", "6 hourly", "every 6 hrs"],
    "every 8 hours": ["q8h", "8 hourly", "every 8 hrs"],
    "every 12 hours": ["q12h", "12 hourly", "every 12 hrs"],
    "as needed": ["prn", "pro re nata", "as required", "when necessary", "sos"],
    
    # Common routes of administration
    "by mouth": ["po", "oral", "orally", "per os"],
    "intravenous": ["iv", "i.v.", "ivp", "iv push"],
    "intramuscular": ["im", "i.m.", "intramuscul"],
    "subcutaneous": ["sc", "s.c.", "subq", "sub q", "subcu"],
    "sublingual": ["sl", "s.l.", "sublingual"],
    "topical": ["top", "topical", "externally"],
    "inhalation": ["inh", "inhale", "breathing"],
    
    # Common prescription instructions
    "with food": ["w/ food", "with meals", "with meal", "ac", "pc"],
    "before meals": ["ac", "a.c.", "before food"],
    "after meals": ["pc", "p.c.", "after food"],
    "with water": ["w/ water", "with h2o"],
    "do not crush": ["no crush", "donot crush", "do not chew", "swallow whole"],
    "take with plenty of water": ["take w/ plenty of h2o", "take w/ plenty of water"],
    "dissolve in water": ["dissolve", "dissolved in water"],
    "until finished": ["until gone", "to completion", "complete course"],
    "shake well": ["shake bottle", "mix well", "agitate"],
}

def apply_medical_dictionary_correction(text):
    """Apply medical dictionary correction to OCR text"""
    if not text:
        return text
        
    words = re.findall(r'\b\w+\b', text.lower())
    corrected_text = text
    
    for word in words:
        # Skip very short words or numbers
        if len(word) < 3 or word.isdigit():
            continue
            
        # Find the best match in our medication dictionary
        best_match = None
        best_score = 0
        best_key = None
        
        for key, aliases in MEDICATION_DICT.items():
            # Check the key itself
            score = fuzz.ratio(word, key.lower())
            if score > best_score and score > 75:  # Threshold of 75%
                best_score = score
                best_match = key
                best_key = key
                
            # Check aliases
            for alias in aliases:
                score = fuzz.ratio(word, alias.lower())
                if score > best_score and score > 75:  # Threshold of 75%
                    best_score = score
                    best_match = key  # Use the standardized term, not the alias
                    best_key = key
        
        if best_match and best_score > 75:
            # Replace the word with the correct spelling, maintaining original case
            pattern = re.compile(re.escape(word), re.IGNORECASE)
            corrected_text = pattern.sub(best_match, corrected_text)
    
    return corrected_text

def preprocess_image(image_path):
    """Simple and effective image preprocessing for prescription OCR."""
    try:
        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            return None
            
        # Simple preprocessing steps for handwritten prescriptions
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        denoised = cv2.fastNlMeansDenoising(gray)
        thresh = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        kernel = np.ones((1, 1), np.uint8)
        processed = cv2.dilate(thresh, kernel, iterations=1)
        
        # Create filename for enhanced image
        base_name = os.path.basename(image_path)
        dir_name = os.path.dirname(image_path)
        enhanced_name = f"enhanced_{base_name}"
        enhanced_path = os.path.join(dir_name, enhanced_name)
        
        # Save enhanced image
        cv2.imwrite(enhanced_path, processed)
        
        return {
            "original": image_path,
            "enhanced": enhanced_path,
            "processed_image": processed
        }
    except Exception as e:
        print(f"Error in image preprocessing: {str(e)}")
        return None

def run_multiple_ocr_passes(image_data):
    """Run multiple OCR passes with different preprocessing settings"""
    try:
        results = []
        
        # Base OCR pass
        if 'enhanced' in image_data and image_data['enhanced']:
            result = ocr_engine.ocr(image_data['enhanced'], cls=True)
            if result and result[0]:
                results.append(result)
        
        # Original image pass
        if 'original' in image_data and image_data['original']:
            result = ocr_engine.ocr(image_data['original'], cls=True)
            if result and result[0]:
                results.append(result)
        
        # If we have no results yet, try different preprocessing on the image
        if len(results) == 0 and 'enhanced' in image_data and image_data['enhanced']:
            # Try inverting the image (helps with white text on dark background)
            enhanced_img = cv2.imread(image_data['enhanced'])
            if enhanced_img is not None:
                inverted = cv2.bitwise_not(enhanced_img)
                inverted_path = os.path.join(os.path.dirname(image_data['enhanced']), "inverted_temp.jpg")
                cv2.imwrite(inverted_path, inverted)
                result = ocr_engine.ocr(inverted_path, cls=True)
                if result and result[0]:
                    results.append(result)
                
                # Clean up
                try:
                    os.remove(inverted_path)
                except:
                    pass
        
        # If still no results, return an empty placeholder result that won't break the system
        if len(results) == 0:
            # Return a placeholder empty result that can be safely processed
            return [{"confidence": 0.0, "empty": True}]
            
        return results
    
    except Exception as e:
        print(f"Error in OCR passes: {str(e)}")
        # Return a placeholder empty result that can be safely processed
        return [{"confidence": 0.0, "empty": True}]

def combine_ocr_results(results):
    """Combine text from multiple OCR passes, handling improved empty results"""
    try:
        if not results:
            return "", None
            
        # Handle empty placeholder result
        if len(results) == 1 and isinstance(results[0], dict) and results[0].get("empty", False):
            return "", 0.0
        
        all_text = []
        confidence_scores = []
        
        for result_set in results:
            if result_set and result_set[0]:
                text = ""
                for line in result_set[0]:
                    if len(line) >= 2:  # Make sure the line has the expected structure
                        text += line[1][0] + "\n"  # text
                        confidence_scores.append(float(line[1][1]))  # confidence
                all_text.append(text)
        
        if not all_text:
            return "", None
            
        # Combine text from all passes, give preference to the first pass
        combined_text = all_text[0]
        
        # Calculate average confidence
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.5
        
        return combined_text.strip(), avg_confidence
    
    except Exception as e:
        print(f"Error combining OCR results: {str(e)}")
        return "", None

def extract_medical_entities(text):
    """Extract medical entities from the text"""
    medications = []
    dosages = []
    frequencies = []
    routes = []
    
    # Use spaCy for entity recognition if available
    if nlp:
        doc = nlp(text)
        for ent in doc.ents:
            if ent.label_ in ["CHEMICAL", "DRUG", "MEDICATION"]:
                # Extract only the medication name without dosage or frequency
                med_name = re.sub(r'\s+\d+\s*\w*\b', '', ent.text) # Remove numbers and units
                med_name = re.sub(r'\b(once|twice|three|four)(\s+times)?\s+(daily|a\s+day)\b', '', med_name, flags=re.IGNORECASE)
                med_name = re.sub(r'\b(every|each)\s+(morning|evening|night|day|hour|hourly)\b', '', med_name, flags=re.IGNORECASE)
                med_name = re.sub(r'\b(qd|bid|tid|qid|prn|od|q\d+h)\b', '', med_name, flags=re.IGNORECASE)
                med_name = med_name.strip()
                if med_name and len(med_name) > 2:  # Ensure we have a reasonable name (not just a unit or directive)
                    medications.append(med_name)
    
    # Extract medications using our dictionary
    for key in MEDICATION_DICT.keys():
        # Skip dosage units, frequency terms, routes, and instructions
        if key in ["milligram", "microgram", "gram", "milliliter", 
                   "once daily", "twice daily", "three times daily", "four times daily",
                   "every morning", "every night", "every hour", "every 4 hours",
                   "every 6 hours", "every 8 hours", "every 12 hours", "as needed",
                   "by mouth", "intravenous", "intramuscular", "subcutaneous",
                   "sublingual", "topical", "inhalation",
                   "with food", "before meals", "after meals", "with water",
                   "do not crush", "take with plenty of water", "dissolve in water",
                   "until finished", "shake well"]:
            continue
            
        if re.search(r'\b' + re.escape(key) + r'\b', text, re.IGNORECASE):
            medications.append(key)
        else:
            # Check aliases, but only for medication items
            for alias in MEDICATION_DICT[key]:
                if re.search(r'\b' + re.escape(alias) + r'\b', text, re.IGNORECASE):
                    medications.append(key)  # Add the standardized term
                    break
    
    # Pattern for dosage (number + unit)
    dosage_pattern = r'\b(\d+[\.\d]*)\s*(mg|mcg|mL|g|mg/mL|mEq|units|tablets?|caps?)\b'
    dosage_matches = re.finditer(dosage_pattern, text, re.IGNORECASE)
    for match in dosage_matches:
        dosages.append(match.group(0))
    
    # Pattern for frequencies
    freq_patterns = [
        r'\b(once|twice|three times|four times)\s+daily\b',
        r'\b(q\.?d|b\.?i\.?d|t\.?i\.?d|q\.?i\.?d)\b',
        r'\b(every|each)\s+(\d+)\s+(hours?|days?)\b',
        r'\b(q)(\d+)(h)\b',
        r'\bprn\b',
        r'\bas needed\b'
    ]
    
    for pattern in freq_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            frequencies.append(match.group(0))
    
    # Pattern for routes of administration
    route_patterns = [
        r'\b(oral(ly)?|by mouth|p\.?o\.)\b',
        r'\b(intravenous|i\.?v\.)\b',
        r'\b(intramuscular|i\.?m\.)\b',
        r'\b(subcutaneous|s\.?c\.|sub-q)\b',
        r'\b(topical(ly)?)\b',
        r'\b(sublingual|s\.?l\.)\b'
    ]
    
    for pattern in route_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            routes.append(match.group(0))
    
    # Clean medication names to remove any residual dosage or frequency info
    clean_medications = []
    for med in medications:
        # Remove dosage and frequency information
        clean_med = re.sub(r'\s+\d+\s*\w*\b', '', med).strip()
        clean_med = re.sub(r'\b(once|twice|three|four)(\s+times)?\s+(daily|a\s+day)\b', '', clean_med, flags=re.IGNORECASE).strip()
        clean_med = re.sub(r'\b(every|each)\s+(morning|evening|night|day|hour|hourly)\b', '', clean_med, flags=re.IGNORECASE).strip()
        clean_med = re.sub(r'\b(qd|bid|tid|qid|prn|od|q\d+h)\b', '', clean_med, flags=re.IGNORECASE).strip()
        
        if clean_med and len(clean_med) > 2:  # Ensure we have a meaningful name
            clean_medications.append(clean_med)
    
    return {
        "medications": list(set(clean_medications)),
        "dosages": list(set(dosages)),
        "frequencies": list(set(frequencies)),
        "routes": list(set(routes))
    }

def process_prescription_with_enhanced_ocr(image_path, output_dir=None):
    """Process a prescription image with enhanced OCR techniques"""
    try:
        # Prepare output paths
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            base_name = os.path.basename(image_path)
            base_name_no_ext = os.path.splitext(base_name)[0]
            enhanced_path = os.path.join(output_dir, f"enhanced_{base_name}")
            results_path = os.path.join(output_dir, f"{base_name_no_ext}_results.json")
        else:
            enhanced_path = None
            results_path = None
        
        # STEP 1: Apply simple image preprocessing
        image_data = preprocess_image(image_path)
        if not image_data:
            return {
                "error": "Failed to preprocess image",
                "raw_text": "",
                "cleaned_text": "",
                "medications": [],
                "dosages": [],
                "frequencies": [],
                "routes": []
            }
        
        # STEP 2: Run multiple OCR passes with different preprocessing
        ocr_results = run_multiple_ocr_passes(image_data)
        
        # STEP 3: Combine text from all OCR passes
        raw_text, confidence = combine_ocr_results(ocr_results)
        
        # Return a basic structure even if text extraction fails
        # This will allow trained images to still work
        if not raw_text:
            return {
                "image_path": image_path,
                "preprocessed_image": image_data.get("enhanced", ""),
                "raw_text": "",
                "cleaned_text": "",
                "medications": [],
                "dosages": [],
                "frequencies": [],
                "routes": [],
                "confidence": confidence if confidence is not None else 50.0,
            }
        
        # STEP 4: Apply medical dictionary correction
        corrected_text = apply_medical_dictionary_correction(raw_text)
        
        # STEP 5: Extract medical entities
        entities = extract_medical_entities(corrected_text)
        
        # Build the results
        results = {
            "image_path": image_path,
            "preprocessed_image": image_data["enhanced"],
            "raw_text": raw_text,
            "cleaned_text": corrected_text,
            "medications": entities["medications"],
            "dosages": entities["dosages"],
            "frequencies": entities["frequencies"],
            "routes": entities["routes"],
            "confidence": float(confidence) * 100 if confidence else 90.0,
        }
        
        # Save results to file if output_dir is provided
        if results_path:
            try:
                with open(results_path, 'w') as f:
                    json.dump(results, f, indent=4)
            except Exception as e:
                print(f"Error saving results: {str(e)}")
        
        return results
        
    except Exception as e:
        print(f"Error in OCR processing: {str(e)}")
        return {
            "error": f"Processing error: {str(e)}",
            "raw_text": "",
            "cleaned_text": "",
            "medications": [],
            "dosages": [],
            "frequencies": [],
            "routes": []
        }

# If running as a script
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced Prescription OCR with PaddleOCR")
    parser.add_argument("--image", "-i", required=True, help="Path to prescription image")
    parser.add_argument("--output", "-o", default="./output", help="Output directory for results")
    
    args = parser.parse_args()
    
    print("===== Enhanced Prescription OCR Processing =====")
    print(f"Processing image: {args.image}")
    
    # Process the prescription
    results = process_prescription_with_enhanced_ocr(args.image, args.output)
    
    # Print the results
    if "error" in results:
        print(f"Error: {results['error']}")
    else:
        print("\n===== OCR Results =====")
        print(f"Preprocessed image: {results['preprocessed_image']}")
        
        print("\nExtracted text:")
        print(results['raw_text'])
        
        print("\nCleaned text:")
        print(results['cleaned_text'])
        
        print("\nDetected medications:")
        if results['medications']:
            for med in results['medications']:
                print(f"- {med}")
        else:
            print("No medications detected")
            
        print("\nDetected dosages:")
        if results['dosages']:
            for dosage in results['dosages']:
                print(f"- {dosage}")
        else:
            print("No dosages detected")
            
        print("\nDetected frequencies:")
        if results['frequencies']:
            for freq in results['frequencies']:
                print(f"- {freq}")
        else:
            print("No frequencies detected")
            
        print("\nDetected routes:")
        if results['routes']:
            for route in results['routes']:
                print(f"- {route}")
        else:
            print("No routes detected")
        
        print(f"\nConfidence: {results['confidence']:.1f}%")
        
        print("\nProcessing complete!")
