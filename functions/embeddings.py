import pickle
from openai.embeddings_utils import (
    get_embedding,
    distances_from_embeddings,
    indices_of_nearest_neighbors_from_distances
)


# Retorna o embedding de uma string usando o cache para evitar computar redundâncias
def embedding_from_string(
    string: str,
    model: str,
    embedding_cache,
    embedding_cache_path
) -> list:
    if (string, model) not in embedding_cache.keys():
        embedding_cache[(string, model)] = get_embedding(string, model)
        with open(embedding_cache_path, "wb") as embedding_cache_file:
            pickle.dump(embedding_cache, embedding_cache_file)
    return embedding_cache[(string, model)]


# Ordenao a lista de termos de acordo com a proximidade da string de busca
def search_pls(full_text, labels, semantic_string, number_of_suggestions, model, embedding_cache, embedding_cache_path):
    semantic_string_embedding = get_embedding(
        semantic_string,
        engine=model
    )

    search_embeddings = [embedding_from_string(pl_string, model=model, embedding_cache=embedding_cache, embedding_cache_path=embedding_cache_path) for pl_string in full_text]
    search_distances = distances_from_embeddings(semantic_string_embedding, search_embeddings, distance_metric="cosine")
    indices_of_distances = indices_of_nearest_neighbors_from_distances(search_distances)

    k_counter = 0
    for i in indices_of_distances:
        current_string = full_text[i]
        if current_string == semantic_string:
            continue
        if k_counter >= number_of_suggestions:
            break
        k_counter += 1
    
        print(
            f"""
        --- Recommendation #{k_counter} (nearest neighbor {k_counter} of {number_of_suggestions}) ---
        String: {current_string}
        Label: {labels[i]}"""        
        )

    return indices_of_distances