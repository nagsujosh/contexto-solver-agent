CURRENT_GAME_ID_FILE = "current_game_id.txt"
LAST_SUCCESSFUL_GAME_ID_FILE = "last_successful_game_id.txt"
RESULTS_DIR = "results"

GLOVE_PATH = "glove.6B.300d.txt"

EMB_CENTER = True
EMB_REMOVE_TOP_K = 3
EMB_FP16 = True

VOCAB_SIZE = 80000
TOP_K_BEST = 5   # top-k tried words tracked for redundancy penalty in MMR

# Probe words played at game start to rapidly identify the answer's semantic domain.
# Chosen to span 15 distinct domains in GloVe embedding space.
N_PROBES = 15
PROBE_WORDS = [
    "animal",   # wildlife, biology, pets
    "food",     # nutrition, cooking, eating
    "music",    # arts, entertainment, sound
    "house",    # buildings, architecture, home
    "water",    # nature, liquid, elements
    "sport",    # physical activity, games
    "money",    # economics, finance, business
    "doctor",   # health, medicine, professional
    "book",     # knowledge, reading, education
    "machine",  # technology, mechanics, devices
    "church",   # religion, spirituality, culture
    "country",  # geography, politics, place
    "child",    # people, family, age
    "rock",     # geology, music, material
    "blue",     # color, attribute, emotion
]

API_BASE = "https://api.contexto.me"
LANG = "en"
RATE_LIMIT_SLEEP = 0.1

BAD_WORDS_CACHE = "bad_words.json"

CORRECTION_BIAS_FILE = "centroid_bias.npy"
CORRECTION_MATRIX_FILE = "centroid_matrix.npy"
CORRECTION_META_FILE = "centroid_meta.json"
CORRECTION_MIN_GAMES = 20
MAX_GUESSES_PER_GAME = 2000
