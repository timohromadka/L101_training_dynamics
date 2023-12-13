import tensorflow as tf
from typing import List, Dict, Tuple

def get_samples_by_index(dataset: tf.data.Dataset, indices: List[int]) -> List[dict]:
    """
    Get samples by their index

    :param dataset:
    :param indices:
    :return:
    """

    # Convert to tensor
    indices_ = tf.convert_to_tensor(indices, dtype=tf.int32)

    samples = list()
    for sample in dataset.filter(lambda x: tf.reduce_any(tf.equal(x['idx'], indices_))).as_numpy_iterator():
        sample['question'], sample['sentence'] = sample['question'].decode(), sample['sentence'].decode()
        samples.append(sample)

        # Early stop
        if len(samples) == len(indices):
            break

    return samples

class InnerDict(TypedDict):
    gold_label_probs: List[float]
    confidence: float
    variability: float
    correctness: float
    forgetfulness: float

TrainingDynamicsDictType = Dict[str, InnerDict]

def get_training_dynamics_by_index(training_dynamics: TrainingDynamicsDictType, indices: List[int]) -> TrainingDynamicsDictType:
    return {k: v for k, v in training_dynamics.items() if int(k) in indices}

def get_samples_and_training_dynamics_by_index(dataset: tf.data.Dataset, training_dynamics: TrainingDynamicsDictType, indices: List[int]):
    return get_samples_by_index(dataset, indices), get_training_dynamics_by_index(training_dynamics, indices)

def save_training_dynamics(tdd_map, reference_set, tag):
    gold_label_probs = tdd_map.gold_labels_probabilities
    confidence = tdd_map.confidence
    variability = tdd_map.variability
    correctness = tdd_map.correctness
    forgetfulness = tdd_map.forgetfulness

    # MUST MAKE ABSOLUTE SURE
    json_dict = {}
    for i, example in enumerate(tqdm(tfds.as_numpy(reference_set), desc='Saving training default dataset to json')):
        idx = convert_numpy(example["idx"])
        json_dict[idx] = {
            "gold_label_probs": convert_numpy(gold_label_probs[i]),
            "confidence": confidence[i],
            "variability": variability[i],
            "correctness": correctness[i],
            "forgetfulness": forgetfulness[i],
        }

    with open(f'results/{tag}.json', 'w') as file:
        # Dump the list of dictionaries to the file
        json.dump(json_dict, file)

def pretty_print_samples(samples: List[dict], title: Optional[str]):
    """
    Print samples as Markdown

    :param samples: Samples to print
    :param title: Title
    :return:
    """

    # Convert samples to markdown
    samples_as_markdown = list()
    for sample in samples:
        samples_as_markdown.append(f"""
 - **Question**: {sample['question']}

   **Sentence**: {sample['sentence']}

   **Label**: {'True' if sample['label'] == 0 else 'False'}


""")
    samples_as_markdown = "\n".join(samples_as_markdown)

    markdown_output = f"""
### {title}
{samples_as_markdown}
"""

    display(Markdown(markdown_output))

def convert_numpy(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, bytes):
        return obj.decode('utf-8')
    else:
        return obj

def extract_by_training_dynamics(datamap: np.ndarray,
                                 n_ambiguous: int = 0,
                                 n_easy2learn: int = 0,
                                 n_hard2learn: int = 0) -> Tuple[list, list, list]:
    """
    Extract indices of different types of samples

    :param datamap: Datamap
    :param n_ambiguous: Number of ambiguous samples to extract
    :param n_easy2learn: Number of easy-to-learn samples to extract
    :param n_hard2learn: Number of hard-to-learn samples to extract
    :return: Three lists of ambiguous, easy-to-learn and hard-to-learn indices
    """

    kdt = KDTree(datamap, metric='euclidean')
    ambiguous = list()
    easy2learn = list()
    hard2learn = list()

    if n_easy2learn:
        _, easy2learn = kdt.query([[0, 1]],
                                  k=n_easy2learn)

    if n_ambiguous:
        _, ambiguous = kdt.query([[1, 0.5]],
                                 k=n_ambiguous)

    if n_hard2learn:
        _, hard2learn = kdt.query([[0, 0]],
                                  k=n_hard2learn)

    return list(ambiguous[0]), list(easy2learn[0]), list(hard2learn[0])

def load_training_dynamics_from_json(path_to_json):
    with open(path_to_json, 'r') as file:
        training_dynamics_dict = json.load(file)

    gold_labels_probabilities = []
    confidences = []
    variabilities = []
    correctnesses = []
    forgetfulnesses = []
    for idx, td_dict in tqdm(training_dynamics_dict.items(), desc=f'Fetching training dynamics from saved json file: {path_to_json}'):
        gold_labels_probabilities.append(td_dict['gold_label_probs'])
        confidences.append(td_dict['confidence'])
        variabilities.append(td_dict['variability'])
        correctnesses.append(td_dict['correctness'])
        forgetfulnesses.append(td_dict['forgetfulness'])

    gold_labels_probabilities = np.array(gold_labels_probabilities)


    return gold_labels_probabilities, confidences, variabilities, correctnesses, forgetfulnesses