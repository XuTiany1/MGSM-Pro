from pathlib import Path
import json, os, random, re
from typing import Dict, List, Set
from collections import defaultdict

#######################
# Helper Functions
#######################

# Function reads all json objects in a folder in order
def read_json_from_dir(directory_path: str):

    objects = []

    # Sort json files
    json_files = sorted(
        (f for f in os.listdir(directory_path)
        if f.endswith(".json") and f[:-5].isdigit())
    )

    for file_name in json_files:
        file_path = os.path.join(directory_path, file_name)

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()

                obj = json.loads(content)
                objects.append(obj)
        except Exception as e:
            print(f"failed reading {file_name}: {e}")
    
    return objects



def fill_template(template_list: list[str],
                  num_dict: Dict[str, List[int]],
                  idx: int,
                  lexicon: Dict[str, List[str]],
                  substitute: Set[str] | None = None) -> str:
    """
    Param
    template: question template with placeholders
    num_dict: mapping from number-varaibles to list of values
    idx: which entry in each number list to use for this instance (I have 10 per template)
    lexicon: the dictionary of tgt language vocab
    substitute: combinations of what to substitute {"numbers", "names"}
    """

    # Substitute being none means I want the original template
    if substitute is None:
        substitute = {}

    pattern = re.compile(r"\{([a-zA-Z0-9_]+),([^}]*)\}")
    chosen = {}
    used = defaultdict(set)

    # Inner function for handling regex matching
    def repl(m: re.Match):
        # Extract variable type with the fallback
        var = m.group(1)
        fallback = m.group(2).strip()

        # Number variables
        if var in num_dict:
            
            # Not changing numbers 
            if "numbers" not in substitute:
                return fallback
            # Changing numbers
            try:
                return str(num_dict[var][idx])
            except IndexError:
                raise ValueError(f"idx{idx} out of range for {var}")

        # Name variables
        if var.startswith("name_"):

            # Not changing names
            if "names" not in substitute:
                return fallback
            
            # Changing names
            base = re.sub(r"\d+$", "", var) #remove the number after the names
            if "male" in base:
                name_type = "name_male"
            elif "female" in base:
                name_type = "name_female"
            elif "city" or "town" in base:
                # Location
                name_type = "name_city"
            elif "mountain" in base:
                name_type = "name_mountain"
            elif "family" in base:
                name_type = "name_family"
            elif "cat" in base:
                name_type = "name_cat"
            elif "dinosaur" in base:
                name_type = "name_dinosaur"
            elif "dragon" in base:
                name_type = "name_dragon"
            else:
                print(F"ERROR: base have no corresponding name base {base}")

            # Avoid duplicates
            if var in chosen:
                return chosen[var]
            # Pool of unused names
            pool = [n for n in lexicon.get(name_type, []) if n not in used[name_type]]
            # No more unsed names
            if not pool:
                raise ValueError(f"No unused {name_type} left for '{var}'")
            # Randomly pick a name
            pick = random.choice(pool)

            # Update the used names
            chosen[var] = pick
            used[name_type].add(pick)
            return pick
        
        return fallback
    # Reset
    chosen = {}
    used = defaultdict(set)

    sub_template_list = []
    for template in template_list:
        sub_template_list.append(pattern.sub(repl, template))

    return sub_template_list


# Function reads all words in the word file (used for my english.json vocab and can be used for other vocabs)
def load_words(names_fp: Path):
    with open(names_fp, encoding="utf-8") as f:
        return json.load(f)

# helper for filename from combination 
def combo_to_filename(combo: Set[str]) -> str:
    if combo == {"numbers"}: 
        return "numbers.json"
    if combo == {"names"}:   
        return "names.json"
    if combo == {"original"}:
        return "original.json"
    # if its a combo, then join together by _ 
    return "_".join(sorted(combo)) + ".json" 


# Fetch the answer that corresponds to the instance_id
def get_answer(sub_pattern: set, 
               question_dictionary: dict,
               instance_id: int):
    # If numbers are changing, then change the final answer
    if "numbers" in sub_pattern:
        number_vars = question_dictionary["variable_number_dictionary"]
        return number_vars.get("answer", [None])[instance_id]

    # Otherwise, don't change the final answer
    else:
        print("ORIGINAL ANWER!")
        return question_dictionary["original_answer"]



#######################
# Hyperparam
#######################
LANGUAGE = "hausa"
DATASET_TYPE = "IC" # ALWAYS USE IC -> SORRY BUT SYMBOLIC IS JUST A SUBSET OF IC

INPUT_FOLDER = f"/home/mila/x/xut/github/afrimgsm-symbolic/template/{LANGUAGE}/{DATASET_TYPE}"
OUTPUT_FOLDER = f"/home/mila/x/xut/github/afrimgsm-symbolic/dataset/{LANGUAGE}/{DATASET_TYPE}"
WORD_FILE = f"/home/mila/x/xut/github/afrimgsm-symbolic/language_data/{LANGUAGE}_words.json"

list_of_sub = [
    {"numbers"},
    {"names"},
    {"numbers", "names"},
    {"original"}
]


#######################
# Main Logic
#######################

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

names_data = load_words(WORD_FILE)
input_objects = read_json_from_dir(INPUT_FOLDER)

for sub_set in list_of_sub:

    output_records: List[dict] = []                      # reset every loop
    outfile = combo_to_filename(sub_set)
    print(f"→ Generating {outfile} with substitutions: {sub_set}")

    for file_id, obj in enumerate(input_objects):

        number_vars = obj["variable_number_dictionary"]

        # Craft for number of instances
        if "numbers" in sub_set and number_vars:
            n_instances = len(next(iter(number_vars.values())))
        if "original" in sub_set:
            n_instances = 1
        else:
            n_instances = 10        # arbitrary: create 10 variants

        for inst_idx in range(n_instances):

            print(obj)

            answer = get_answer(sub_pattern=sub_set,
                                question_dictionary=obj,
                                instance_id=inst_idx)


            try:

                template_list = [obj["ic_question_template"], obj["question_template"]]

                sub_template_list = fill_template(
                        template_list = template_list,
                        num_dict = number_vars,
                        idx = inst_idx,
                        lexicon = names_data,
                        substitute = sub_set
                )

                ic_question = sub_template_list[0]
                symbolic_quesiton = sub_template_list[1]

                print(f"ic_question {ic_question}")
                print(f"symbolic_quesiton {symbolic_quesiton}")


                #if DATASET_TYPE == "IC":
                #    template = obj["ic_question_template"]
                #elif DATASET_TYPE == "symbolic":
                #    template = obj["question_template"]
                #else:
                #    template = None
                #    print("[ERROR] DATASET_TYPE is invalid")

                #q = fill_template(
                #        template = template,
                #        num_dict = number_vars,
                #        idx = inst_idx,
                #        lexicon = names_data,
                #        substitute = sub_set
                #    )

                output_records.append({
                    "id": file_id,
                    "instance": inst_idx,
                    "symbolic_question": symbolic_quesiton,
                    "ic_question" : ic_question,
                    "answer": answer,
                    "equation": obj.get("question_solution"),
                    "conditions": obj.get("conditions"),
                    "original_question": obj.get("original_question"),
                    "original_answer": obj.get("original_answer"),
                })

            except Exception as e:
                print(f"[ERROR] file {file_id} inst {inst_idx}: {e}")

    # write this combination’s file
    out_path = os.path.join(OUTPUT_FOLDER, outfile)
    with open(out_path, "w", encoding="utf-8") as f:
        body = ",\n".join(json.dumps(r, ensure_ascii=False) for r in output_records)
        f.write("[\n" + body + "\n]\n")

    print(f" {len(output_records)} records → {out_path}")


