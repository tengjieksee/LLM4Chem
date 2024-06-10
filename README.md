# LlaSMol
This is the official code repository for the paper *LlaSMol: Advancing Large Language Models for Chemistry with a Large-Scale, Comprehensive, High-Quality Instruction Tuning Dataset*.

- Paper: https://arxiv.org/abs/2402.09391
- Page: https://osu-nlp-group.github.io/LLM4Chem
- Dataset: https://huggingface.co/datasets/osunlp/SMolInstruct
- Models:
  - LlaSMol-Galactica-6.7B: [https://huggingface.co/osunlp/LlaSMol-Galactica-6.7B](https://huggingface.co/osunlp/LlaSMol-Galactica-6.7B)
  - LlaSMol-Llama2-7B: [https://huggingface.co/osunlp/LlaSMol-Llama2-7B](https://huggingface.co/osunlp/LlaSMol-Llama2-7B)
  - LlaSMol-CodeLlama-7B: [https://huggingface.co/osunlp/LlaSMol-CodeLlama-7B](https://huggingface.co/osunlp/LlaSMol-CodeLlama-7B)
  - LlaSMol-Mistral-7B: [https://huggingface.co/osunlp/LlaSMol-Mistral-7B](https://huggingface.co/osunlp/LlaSMol-Mistral-7B)

## Requirements
```
accelerate==0.24.1
aiofiles==23.2.1
aiohttp==3.8.6
aiosignal==1.3.1
altair==5.1.2
annotated-types==0.6.0
anthropic==0.24.0
anyio==3.7.1
appdirs==1.4.4
async-timeout==4.0.3
attrs==23.1.0
beautifulsoup4==4.9.0
bitsandbytes==0.41.3.post2
certifi==2023.7.22
charset-normalizer==3.3.0
click==8.1.7
conda-pack @ file:///home/conda/feedstock_root/build_artifacts/conda-pack_1691435290924/work
contourpy==1.1.1
cycler==0.12.1
dataclasses-json==0.5.14
datasets==2.14.5
dill==0.3.7
distro==1.9.0
docker-pycreds==0.4.0
exceptiongroup==1.1.3
fastapi==0.103.2
ffmpy==0.3.1
filelock==3.12.4
fire==0.5.0
fonttools==4.43.1
frozenlist==1.4.0
fsspec==2023.6.0
gitdb==4.0.10
GitPython==3.1.37
gradio==3.47.1
gradio_client==0.6.0
greenlet==3.0.3
h11==0.14.0
httpcore==0.18.0
httpx==0.25.0
huggingface-hub==0.17.3
idna==3.4
importlib-resources==6.1.0
Jinja2==3.1.2
jsonpatch==1.33
jsonpointer==2.4
jsonschema==4.19.1
jsonschema-specifications==2023.7.1
kiwisolver==1.4.5
langchain==0.0.275
langchain-core==0.2.1
langchain-text-splitters==0.2.0
langsmith==0.0.92
MarkupSafe==2.1.3
marshmallow==3.21.2
matplotlib==3.8.0
mkl-fft @ file:///croot/mkl_fft_1695058164594/work
mkl-random @ file:///croot/mkl_random_1695059800811/work
mkl-service==2.4.0
molbloom==2.2.1
multidict==6.0.4
multiprocess==0.70.15
mypy-extensions==1.0.0
numexpr==2.10.0
numpy==1.26.1
openai==0.27.8
orjson==3.10.3
packaging==23.2
pandas==2.1.1
pathtools==0.1.2
peft==0.7.0
Pillow==10.1.0
protobuf==4.24.4
psutil==5.9.6
PubChemPy==1.0.4
pyarrow==13.0.0
pydantic==2.4.2
pydantic_core==2.10.1
pydub==0.25.1
pynvml==11.5.0
pyparsing==3.1.1
python-dateutil==2.8.2
python-multipart==0.0.6
pytz==2023.3.post1
PyYAML==6.0.1
rdchiral==1.1.0
rdkit==2023.9.5
referencing==0.30.2
regex==2023.10.3
requests==2.31.0
rpds-py==0.10.6
RXN4Chemistry==1.14.0
safetensors==0.4.0
scipy @ file:///croot/scipy_1701295040508/work/dist/scipy-1.11.4-cp39-cp39-linux_x86_64.whl#sha256=1f9b757a21b482354a59770c1b025701f2228439c621dc0894a792120398a6e9
selfies==2.1.1
semantic-version==2.10.0
sentencepiece==0.1.99
sentry-sdk==1.32.0
setproctitle==1.3.3
six==1.16.0
smmap==5.0.1
sniffio==1.3.0
soupsieve==2.5
SQLAlchemy==2.0.30
starlette==0.27.0
tenacity==8.3.0
termcolor==2.3.0
tokenizers==0.14.1
toolz==0.12.0
torch==1.13.1
tqdm==4.66.1
transformers==4.34.1
typing-inspect==0.9.0
typing_extensions @ file:///croot/typing_extensions_1705005625920/work
tzdata==2023.3
urllib3==2.0.6
uvicorn==0.23.2
wandb==0.15.12
websockets==11.0.3
xxhash==3.4.1
yarl==1.9.2
zipp==3.17.0
```



## Tasks and Examples

Our models are trained on 4 types of 14 essential tasks, which, along with examples, are listed below.

### Name Conversion
**Name Conversion - IUPAC to Molecular Formula**
```
Query: What is the molecular formula of the compound with this IUPAC name <IUPAC> 2,5-diphenyl-1,3-oxazole </IUPAC> ?

Response: <MOLFORMULA> C15H11NO </MOLFORMULA>
```

**Name Conversion - IUPAC to SMILES**
```
Query: Could you provide the SMILES for <IUPAC> 4-ethyl-4-methyloxolan-2-one </IUPAC> ?

Response: Of course. It's <SMILES> CCC1(C)COC(=O)C1 </SMILES> .
```

**Name Conversion - SMILES to Molecular Formula**
```
Query: Given the SMILES representation <SMILES> S=P1(N(CCCl)CCCl)NCCCO1 </SMILES>, what would be its molecular formula?

Response: It is <MOLFORMULA> C7H15Cl2N2OPS </MOLFORMULA> .
```

**Name Conversion - SMILES to IUPAC**
```
Query: Translate the given SMILES formula of a molecule <SMILES> CCC(C)C1CNCCCNC1 </SMILES> into its IUPAC name.

Response: <IUPAC> 3-butan-2-yl-1,5-diazocane </IUPAC>
```

### Property Prediction
**Property Prediction - ESOL**
```
Query: How soluble is <SMILES> CC(C)Cl </SMILES> ?

Response: Its log solubility is <NUMBER> -1.41 </NUMBER> mol/L.
```

**Property Prediction - LIPO**
```
Query: Predict the octanol/water distribution coefficient logD under the circumstance of pH 7.4 for <SMILES> NC(=O)C1=CC=CC=C1O </SMILES> .

Response: <NUMBER> 1.090 </NUMBER>
```

**Property Prediction - BBBP**
```
Query: Is blood-brain barrier permeability (BBBP) a property of <SMILES> CCNC(=O)/C=C/C1=CC=CC(Br)=C1 </SMILES> ?

Response: <BOOLEAN> Yes </BOOLEAN>
```

**Property Prediction - Clintox**
```
Query: Is <SMILES> COC[C@@H](NC(C)=O)C(=O)NCC1=CC=CC=C1 </SMILES> toxic?

Response: <BOOLEAN> No </BOOLEAN>
```

**Property Prediction - HIV**
```
Query: Can <SMILES> CC1=CN(C2C=CCCC2O)C(=O)NC1=O </SMILES> serve as an inhibitor of HIV replication?

Response: <BOOLEAN> No </BOOLEAN>
```

**Property Prediction - SIDER**
```
Query: Are there any known side effects of <SMILES> CC1=CC(C)=C(NC(=O)CN(CC(=O)O)CC(=O)O)C(C)=C1Br </SMILES> affecting the heart?

Response: <BOOLEAN> No </BOOLEAN>
```

### Molecule Description

**Molecule Captioning**
```
Query: Describe this molecule: <SMILES> CCOC(=O)C1=CN=CN1[C@H](C)C1=CC=CC=C1 </SMILES>

Response: The molecule is an imidazole derivative with short-acting sedative, hypnotic, and general anesthetic properties. Etomidate appears to have gamma-aminobutyric acid (GABA) like effects, mediated through GABA-A receptor. The action enhances the inhibitory effect of GABA on the central nervous system by causing chloride channel opening events which leads to membrane hyperpolarization.
```

**Molecule Generation**

It is not required to use tags to wrap up the input description.

```
Query: Give me a molecule that satisfies the conditions outlined in the description: The molecule is a member of the class of tripyrroles that is a red-coloured pigment with antibiotic properties produced by Serratia marcescens. It has a role as an antimicrobial agent, a biological pigment, a bacterial metabolite, an apoptosis inducer and an antineoplastic agent. It is a tripyrrole, an aromatic ether and a ring assembly.

Response: Here is a potential molecule: <SMILES> CCCCCC1=C(C)NC(/C=C2\N=C(C3=CC=CN3)C=C2OC)=C1 </SMILES>
```

### Chemical Reaction

**Forward Synthesis**
```
Query: <SMILES> NC1=CC=C2OCOC2=C1.O=CO </SMILES> Based on the reactants and reagents given above, suggest a possible product.

Response: A possible product can be <SMILES> O=CNC1=CC=C2OCOC2=C1 </SMILES> .
```

**Retrosynthesis**
```
Query: Identify possible reactants that could have been used to create the specified product. <SMILES> CC1=CC=C(N)N=C1N </SMILES>

Response: <SMILES> CC(C#N)CCC#N.N </SMILES>
```

## Training

If you need to fine-tune a base model on SMolInstruct, please first clone this repo to your machine, and `cd` to the folder, then use the following command.

```bash
MODELNAME=LlaSMol-Mistral-7B && CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch finetune.py --data_path osunlp/SMolInstruct --base_model mistralai/Mistral-7B-v0.1 --wandb_project LlaSMol --wandb_run_name $MODELNAME --wandb_log_model true ---output_dir checkpoint/$MODELNAME
```

The above is an example command for fine-tuning Mistral with LoRA, using 4 GPUs. If with other base models, the lora settings (e.g., `lora_target_modules`) might need to be modified accordingly.



## Usage

Clone this repo to your machine, and `cd` to the folder.

### Generation

You could use the following code to query the models with your questions.

```python
from generation import LlaSMolGeneration

generator = LlaSMolGeneration('osunlp/LlaSMol-Mistral-7B')
generator.generate('Can you tell me the IUPAC name of <SMILES> C1CCOC1 </SMILES> ?')
```

**Note**: 
1. In the input query, please use corresponding tags to wrap up specific content. 
    - SMILES representation: `<SMILES> ... </SMILES>`
    - IUPAC name: `<IUPAC> ... </IUPAC>`
    
    Other tags may appear in models' responses:
    - Molecular formula: `<MOLFORMULA> ... </MOLFORMULA>`
    - Number: `<NUMBER> ... </NUMBER>`
    - Boolean: `<BOOLEAN> ... </BOOLEAN>`

    Please see the examples in [the above section](#tasks-and-examples).

2. The code would canonicalize SMILES string automatically, as long as it is wrapped in `<SMILES> ... </SMILES>`.

### Evaluation on SMolInstruct

#### Step 1. Generate responses for samples

Use the following command to apply LlaSMol models to generate responses for samples in SmolInstruct.

```bash
python generate_on_dataset.py --model_name osunlp/LlaSMol-Mistral-7B --output_dir eval/LlaSMol-Mistral-7B/output 
```

By default, it generates for all the tasks. You could also specify tasks by adding argument like `--tasks "['forward_synthesis','retrosynthesis']"`.
If not setting `tasks`, the script will generate for all the tasks in SMolInstruct.

#### Step 2. Extract predicted answer from model outputs

Use the command to extract predicted answers from model's output, and store them in the `pred` domains. By default, it extract the part between the corresponding tags (e.g., `<SMILES> ... </SMILES>`). If the tags are missing or incomplete, the extracted answer will be empty and regarded as "no answer" in metric calculation.

```bash
python extract_prediction.py --output_dir eval/LlaSMol-Mistral-7B/output --prediction_dir eval/LlaSMol-Mistral-7B/prediction
```

By default, it extracts predicted answers for all the tasks. It skips task if its output file is not found. You could also specify tasks like  `--tasks "['forward_synthesis','retrosynthesis']"`.

#### Step 3. Calculate metrics

Use the following command to compute metrics for all the tasks.

```bash
python compute_metrics.py --prediction_dir eval/LlaSMol-Mistral-7B/prediction
```

By default, it extracts predicted answers for all the tasks. It skips task if its output file is not found. You could also specify tasks like  `--tasks "['forward_synthesis','retrosynthesis']"`.

## Citation
If our paper or related resources prove valuable to your research, we kindly ask for citation. Please feel free to contact us with any inquiries.
```
@article{yu2024llasmol,
    title={LlaSMol: Advancing Large Language Models for Chemistry with a Large-Scale, Comprehensive, High-Quality Instruction Tuning Dataset},
    author={Botao Yu and Frazier N. Baker and Ziqi Chen and Xia Ning and Huan Sun},
    journal={arXiv preprint arXiv:2402.09391},
    year={2024}
}
```
