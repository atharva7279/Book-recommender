import pandas as pd
import numpy as np

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

import gradio as gr

# ---------------------------
# Load data
# ---------------------------

books = pd.read_csv("books_with_emotions.csv")

books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
books["large_thumbnail"] = np.where(
    books["thumbnail"].isna(),
    "cover-not-found.jpg",
    books["large_thumbnail"],
)

# ---------------------------
# Load embedding model
# ---------------------------

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ---------------------------
# Load existing Chroma DB
# ---------------------------

db_books = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings,
)

# ---------------------------
# Recommendation function
# ---------------------------

def retrieve_semantic_recommendations(
    query: str,
    category: str = "All",
    tone: str = "All",
    initial_top_k: int = 50,
    final_top_k: int = 16,
):

    recs = db_books.similarity_search(query, k=initial_top_k)

    books_list = [
        int(rec.page_content.strip('"').split()[0])
        for rec in recs
    ]

    book_recs = books[books["isbn13"].isin(books_list)]

    if category != "All":
        book_recs = book_recs[
            book_recs["simple_categories"] == category
        ]

    if tone == "Happy":
        book_recs = book_recs.sort_values("joy", ascending=False)

    elif tone == "Surprising":
        book_recs = book_recs.sort_values("surprise", ascending=False)

    elif tone == "Angry":
        book_recs = book_recs.sort_values("anger", ascending=False)

    elif tone == "Suspenseful":
        book_recs = book_recs.sort_values("fear", ascending=False)

    elif tone == "Sad":
        book_recs = book_recs.sort_values("sadness", ascending=False)

    return book_recs.head(final_top_k)

# ---------------------------
# Gradio callback
# ---------------------------

def recommend_books(query, category, tone):

    recommendations = retrieve_semantic_recommendations(
        query,
        category,
        tone,
    )

    results = []

    for _, row in recommendations.iterrows():

        description = str(row["description"])

        words = description.split()
        short_description = " ".join(words[:30]) + "..."

        authors = str(row["authors"]).split(";")

        if len(authors) == 2:
            author_string = f"{authors[0]} and {authors[1]}"

        elif len(authors) > 2:
            author_string = (
                ", ".join(authors[:-1])
                + ", and "
                + authors[-1]
            )

        else:
            author_string = authors[0]

        caption = (
            f"{row['title']}\n"
            f"by {author_string}\n\n"
            f"{short_description}"
        )

        results.append(
            (row["large_thumbnail"], caption)
        )

    return results

# ---------------------------
# UI
# ---------------------------

categories = ["All"] + sorted(
    books["simple_categories"].dropna().unique()
)

tones = [
    "All",
    "Happy",
    "Surprising",
    "Angry",
    "Suspenseful",
    "Sad",
]

with gr.Blocks(theme=gr.themes.Glass()) as dashboard:

    gr.Markdown("# 📚 Semantic Book Recommender")

    with gr.Row():

        user_query = gr.Textbox(
            label="Describe the book you want",
            placeholder="e.g. A story about forgiveness and hope",
        )

        category_dropdown = gr.Dropdown(
            choices=categories,
            value="All",
            label="Category",
        )

        tone_dropdown = gr.Dropdown(
            choices=tones,
            value="All",
            label="Emotion",
        )

    submit_button = gr.Button("Find Recommendations")

    output = gr.Gallery(
        label="Recommended Books",
        columns=4,
        height="auto",
    )

    submit_button.click(
        fn=recommend_books,
        inputs=[
            user_query,
            category_dropdown,
            tone_dropdown,
        ],
        outputs=output,
    )

if __name__ == "__main__":
    dashboard.launch(
        server_name="127.0.0.1",
        inbrowser=True,
    )