CURRENT_GAME_ID_FILE = "current_game_id.txt"
LAST_SUCCESSFUL_GAME_ID_FILE = "last_successful_game_id.txt"
RESULTS_DIR = "results"

# Path to the local GloVe 6B 300d embedding file
GLOVE_PATH = "glove.6B.300d.txt"

EMB_CENTER = True           # subtract mean before whitening
EMB_REMOVE_TOP_K = 3        # remove top-k principal directions (0 to skip)
EMB_FP16 = True             # store normalized embeddings as float16 (memory win)

VOCAB_SIZE = 80000
SEED_COUNT = 5
NEIGHBOR_K = 64            # wider local pool helps exploration/exploitation
INITIAL_TEMPERATURE = 0.9
TEMPERATURE_DECAY = 0.97
MIN_TEMPERATURE = 0.25
EXPLOIT_BASE = 0.55
UCB_ALPHA = 1.3

# --- Clustered exploration ---
KMEANS_K = 512              # cluster count for exploration UCB
KMEANS_RANDOM_STATE = 42
CLUSTER_SAMPLE_K = 512      # candidate words sampled from target cluster

# --- Probing the space early (broad categories) ---
EARLY_PROBES = [
    "animal","bird","fish","mammal","reptile","insect",
    "color","shape","size","weight","length","material",
    "tool","device","machine","vehicle","computer","phone","robot",
    "food","fruit","vegetable","meat","drink","dessert",
    "person","child","adult","teacher","doctor","artist","engineer","athlete",
    "city","country","river","mountain","ocean","island","building","school",
    "music","movie","book","game","sport","language","science","history",
    "money","market","bank","company","job","office","factory",
    "house","kitchen","bathroom","garden","road","bridge","airport","station"
]
EARLY_PROBE_TURNS = 8

API_BASE = "https://api.contexto.me"
LANG = "en"
RATE_LIMIT_SLEEP = 0.1

BAD_WORDS_CACHE = "bad_words.json"
