import json
import random
from pathlib import Path
import sys

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config import load_config
from src.data.generators.verbs import VerbGenerator
from src.data.generators.syntax import SyntaxGenerator
from src.data.generators.cases import CaseGenerator

class MasterGenerator:
    """Master class that combines all sub-generators."""
    
    def __init__(self):
        self.v_gen = VerbGenerator()
        self.s_gen = SyntaxGenerator()
        self.c_gen = CaseGenerator()

    def generate_all(self):
        """Combines examples from all topics (A1 and A2)."""
        dataset = []
        
        # Verb topics
        dataset.extend(self.v_gen.generate_praesens(1500))
        dataset.extend(self.v_gen.generate_haben_sein_praesens(1500))
        dataset.extend(self.v_gen.generate_perfekt_aux(1500))
        dataset.extend(self.v_gen.generate_partizip_forms(1500))
        dataset.extend(self.v_gen.generate_modal_verbs(1000))
        dataset.extend(self.v_gen.generate_separable_verbs(1500))
        dataset.extend(self.v_gen.generate_reflexive_verbs(1000))
        dataset.extend(self.v_gen.generate_praeteritum_essentials(1000))
        dataset.extend(self.v_gen.generate_imperativ(1000))
        dataset.extend(self.v_gen.generate_strong_verbs_praesens(500))
        
        # Syntax topics
        dataset.extend(self.s_gen.generate_inversion(1500))
        dataset.extend(self.s_gen.generate_nebensatz_weil(1000))
        dataset.extend(self.s_gen.generate_questions(1000))
        dataset.extend(self.s_gen.generate_nebensatz_dass_wenn(1000))
        dataset.extend(self.s_gen.generate_negation(1000))
        
        # Case topics (all four cases: Nominativ, Genitiv, Dativ, Akkusativ)
        dataset.extend(self.c_gen.generate_nominativ(1000))
        dataset.extend(self.c_gen.generate_akkusativ_masculine(1500))
        dataset.extend(self.c_gen.generate_article_required_akkusativ(500))
        dataset.extend(self.c_gen.generate_genitiv(1200))
        dataset.extend(self.c_gen.generate_dativ(1000))
        dataset.extend(self.c_gen.generate_prepositions_akk_dat(1000))
        dataset.extend(self.c_gen.generate_adjective_endings(1000))
        dataset.extend(self.c_gen.generate_possessive_pronouns(1000))
        dataset.extend(self.c_gen.generate_komparation(1000))
        dataset.extend(self.c_gen.generate_fixed_prepositions(1000))
        
        random.shuffle(dataset)
        return dataset

    def save(self, data, path="data/train.jsonl"):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, 'w', encoding='utf-8') as f:
            for entry in data:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        print(f"🚀 Generated {len(data)} examples in {path}")

if __name__ == "__main__":
    config = load_config()
    master = MasterGenerator()
    data = master.generate_all()

    # Split into training and validation (90/10)
    split = int(len(data) * 0.9)
    master.save(data[:split], config.data.train_path)
    master.save(data[split:], config.data.val_path)

    # Rebuild vocab from new train/val so all generated words are in the dictionary
    from src.tokenizer.build_vocab import build_vocab
    vocab = build_vocab()
    vocab_path = Path(__file__).resolve().parent.parent / "tokenizer" / "vocab.json"
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    print(f"✅ vocab.json rebuilt — {len(vocab)} tokens")
