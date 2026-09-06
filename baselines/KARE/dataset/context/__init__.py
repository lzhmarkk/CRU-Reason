from baselines.KARE.dataset.context.community import partition_and_summarize_cached
from baselines.KARE.dataset.context.argument import get_argument_context
from baselines.KARE.dataset.context.personal_graph import get_patient_graph


def KARE_context(sample_dataset, embeddings, g, model, task, output_dir):
    communities, summary2nodes, summary2emb = partition_and_summarize_cached(g, model, output_dir)

    patient_nodes = get_patient_graph(task, sample_dataset, g, output_dir)

    selected_summaries_all = get_argument_context(patient_nodes, embeddings, summary2nodes, summary2emb, task, output_dir)

    return selected_summaries_all
