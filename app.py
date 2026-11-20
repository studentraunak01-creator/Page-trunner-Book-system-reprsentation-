from pathlib import Path
import json
import random
from textwrap import dedent

import pandas as pd
import streamlit as st

from src.recommender import (
    load_processed_books,
    build_tfidf_matrix,
    recommend_books,
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="PageTurnner | Discover your next read",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

PROJECT_ROOT = Path(__file__).resolve().parent
READING_LIST_PATH = PROJECT_ROOT / "user_data" / "reading_list.json"

GENRE_KEYWORDS = {
    "Fantasy": ["fantasy", "magic", "wizard", "witch", "dragon", "myth"],
    "Romance": ["romance", "love", "relationships", "chick lit"],
    "Mystery & Crime": ["mystery", "crime", "detective", "thriller", "suspense"],
    "Science Fiction": ["science fiction", "sci fi", "dystopia", "space", "aliens"],
    "Horror": ["horror", "ghost", "vampire", "zombie", "paranormal"],
    "Historical": ["historical", "history", "historical fiction", "war"],
    "Young Adult": ["young adult", "ya", "teen", "coming of age"],
    "Classics": ["classics", "classic literature", "literary fiction"],
    "Nonfiction": ["nonfiction", "biography", "memoir", "science", "psychology"],
}
GENRE_OPTIONS = ["Any genre", *GENRE_KEYWORDS.keys()]


# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown(
    dedent("""
    <style>
    :root {
        --ink: #23372f;
        --muted: #6d786f;
        --cream: #f6f4ee;
        --paper: #fffefa;
        --green: #315b45;
        --green-dark: #20392f;
        --line: #e7e4da;
        --gold: #c49a53;
    }

    .stApp, [data-testid="stAppViewContainer"] {
        background: var(--cream);
        color: var(--ink);
    }

    [data-testid="stHeader"] {
        background: rgba(246, 244, 238, 0.88);
    }

    [data-testid="stMainBlockContainer"] {
        max-width: 1320px;
        padding-top: 1.6rem;
        padding-bottom: 3.5rem;
    }

    [data-testid="stSidebar"] {
        background: var(--green-dark);
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    [data-testid="stSidebar"] * {
        color: #f5f4ec;
    }

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: #d7e0d8;
    }

    [data-testid="stSidebar"] .stRadio label {
        padding: 0.30rem 0;
    }

    h1, h2, h3, h4 {
        color: var(--ink) !important;
        letter-spacing: -0.035em;
    }

    [data-testid="stMarkdownContainer"] p,
    [data-testid="stCaptionContainer"] {
        color: var(--muted);
    }

    .brand-mark {
        font-size: 1.9rem;
        font-weight: 800;
        letter-spacing: -0.06em;
        color: #ffffff;
        margin-bottom: 0;
    }

    .brand-subtitle {
        font-size: 0.78rem;
        color: #c8d7cc;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-top: 0.1rem;
    }

    .hero {
        position: relative;
        overflow: hidden;
        background: linear-gradient(115deg, #e5ede3 0%, #f0e8d7 100%);
        border: 1px solid #dce4d9;
        border-radius: 26px;
        padding: 2.8rem 3rem;
        margin: 0.5rem 0 1.5rem 0;
    }

    .hero:after {
        content: "📖";
        position: absolute;
        right: 5%;
        bottom: -25px;
        font-size: 9rem;
        opacity: 0.16;
        transform: rotate(-12deg);
    }

    .hero-kicker {
        color: #53715c;
        font-size: 0.76rem;
        font-weight: 800;
        letter-spacing: 0.17em;
        text-transform: uppercase;
    }

    .hero-title {
        color: #20392f;
        font-size: clamp(2.2rem, 5vw, 3.6rem);
        line-height: 1.06;
        font-weight: 850;
        letter-spacing: -0.065em;
        margin: 0.8rem 0 0.7rem 0;
        max-width: 680px;
    }

    .hero-copy {
        color: #52665a;
        font-size: 1.04rem;
        line-height: 1.65;
        max-width: 610px;
    }

    .section-kicker {
        color: #8b754e;
        font-size: 0.73rem;
        font-weight: 800;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        margin-bottom: 0.25rem;
    }

    .soft-panel {
        background: var(--paper);
        border: 1px solid var(--line);
        border-radius: 18px;
        padding: 1.15rem 1.25rem;
    }

    [data-testid="stMetric"] {
        background: var(--paper);
        border: 1px solid var(--line);
        border-radius: 16px;
        padding: 1rem 1.15rem;
    }

    [data-testid="stMetricLabel"] {
        color: #718075 !important;
    }

    [data-testid="stMetricValue"] {
        color: var(--ink) !important;
    }

    div[data-baseweb="input"] > div,
    div[data-baseweb="select"] > div {
        background: #fffefa !important;
        border: 1px solid #dedfd5 !important;
        border-radius: 11px !important;
    }

    div[data-testid="stButton"] button {
        border-radius: 11px;
        font-weight: 650;
        border: 1px solid #d9dfd5;
        min-height: 2.55rem;
    }

    div[data-testid="stButton"] button[kind="primary"] {
        background: var(--green);
        color: #ffffff;
        border: 1px solid var(--green);
    }

    div[data-testid="stButton"] button[kind="primary"]:hover {
        background: #244633;
        border-color: #244633;
        color: #ffffff;
    }

    [data-testid="stVerticalBlockBorderWrapper"] {
        background: var(--paper);
        border: 1px solid var(--line);
        border-radius: 17px;
        padding: 0.5rem;
        transition: transform 150ms ease, box-shadow 150ms ease;
    }

    [data-testid="stVerticalBlockBorderWrapper"]:hover {
        box-shadow: 0 8px 24px rgba(35, 55, 47, 0.08);
        transform: translateY(-2px);
    }

    [data-testid="stImage"] img {
        border-radius: 9px;
    }

    [data-testid="stProgressBar"] > div > div > div {
        background: var(--green);
    }

    [data-testid="stTabs"] button {
        color: var(--green-dark);
        font-weight: 650;
    }

    hr {
        border-color: var(--line);
    }

    .small-note {
        color: #78847a;
        font-size: 0.82rem;
    }

    .footer {
        color: #7b857b;
        text-align: center;
        font-size: 0.82rem;
        padding: 1rem 0 0.2rem 0;
    }
    </style>
    """),
    unsafe_allow_html=True,
)


# ============================================================
# LOCAL READING LIST STORAGE
# ============================================================
def load_reading_list():
    """Read saved book IDs from the local JSON file."""
    try:
        if READING_LIST_PATH.exists():
            with READING_LIST_PATH.open("r", encoding="utf-8") as file:
                saved = json.load(file)
            if isinstance(saved, list):
                return [str(item) for item in saved]
    except (OSError, json.JSONDecodeError):
        pass
    return []


def save_reading_list(book_ids):
    """Save book IDs locally so the shelf survives app restarts."""
    READING_LIST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with READING_LIST_PATH.open("w", encoding="utf-8") as file:
        json.dump(sorted(set(str(item) for item in book_ids)), file, indent=2)


if "reading_list" not in st.session_state:
    st.session_state.reading_list = load_reading_list()


# ============================================================
# LOAD BOOKS AND BUILD THE MODEL ONCE
# ============================================================
@st.cache_resource(show_spinner="Preparing your library and recommendation model...")
def load_recommendation_engine():
    books = load_processed_books().copy()
    _tfidf, tfidf_matrix = build_tfidf_matrix(books)

    # Normalize fields used by the interface.
    for column in ["title", "authors", "tags_text", "image_url"]:
        if column not in books.columns:
            books[column] = ""
        books[column] = books[column].fillna("").astype(str)

    if "book_id" in books.columns:
        books["book_id"] = books["book_id"].astype(str)

    for column in ["average_rating", "ratings_count", "original_publication_year"]:
        if column in books.columns:
            books[column] = pd.to_numeric(books[column], errors="coerce")

    return books, tfidf_matrix


try:
    books, tfidf_matrix = load_recommendation_engine()
except Exception as error:
    st.error(f"PageTurnner could not load the book collection: {error}")
    st.info("Check that preprocessing has been run and books_processed.csv exists.")
    st.stop()


# ============================================================
# HELPER FUNCTIONS
# ============================================================
def filter_by_genre(frame, genre):
    if genre == "Any genre" or "tags_text" not in frame.columns:
        return frame

    keywords = GENRE_KEYWORDS[genre]
    tags = frame["tags_text"].fillna("").astype(str).str.lower()
    mask = pd.Series(False, index=frame.index)
    for keyword in keywords:
        mask = mask | tags.str.contains(keyword, regex=False, na=False)
    return frame[mask]


def book_identifier(book):
    value = book.get("book_id", "")
    return str(value) if pd.notna(value) else str(book.get("title", ""))


def toggle_saved(book):
    book_id = book_identifier(book)
    saved = list(st.session_state.reading_list)

    if book_id in saved:
        saved.remove(book_id)
    else:
        saved.append(book_id)

    st.session_state.reading_list = saved
    save_reading_list(saved)
    st.rerun()


def render_book_grid(frame, key_prefix="books", show_similarity=False):
    """Display a responsive four-column book grid with save buttons."""
    if frame.empty:
        st.info("No books match these filters. Try changing your selection.")
        return

    for start in range(0, len(frame), 4):
        row = frame.iloc[start:start + 4]
        columns = st.columns(4, gap="medium")

        for column, (_, book) in zip(columns, row.iterrows()):
            with column:
                with st.container(border=True):
                    image_url = str(book.get("image_url", "")).strip()
                    if image_url.startswith("http"):
                        st.image(image_url, width=145)
                    else:
                        st.markdown("<div style='font-size:4rem;text-align:center;padding:1rem'>📖</div>", unsafe_allow_html=True)

                    title = str(book.get("title", "Untitled"))
                    author = str(book.get("authors", "Unknown author"))
                    st.markdown(f"**{title}**")
                    st.caption(f"by {author}")

                    rating = book.get("average_rating")
                    if pd.notna(rating):
                        st.write(f"⭐ {float(rating):.2f} / 5")

                    year = book.get("original_publication_year")
                    if pd.notna(year):
                        st.caption(f"Published {int(year)}")

                    if show_similarity and pd.notna(book.get("similarity_score")):
                        score = float(book["similarity_score"])
                        score = max(0.0, min(score, 1.0))
                        st.progress(score)
                        st.caption(f"Content match · {score:.0%}")

                    book_id = book_identifier(book)
                    is_saved = book_id in st.session_state.reading_list
                    button_label = "♥ Saved to shelf" if is_saved else "♡ Save to shelf"
                    if st.button(
                        button_label,
                        key=f"{key_prefix}_save_{book_id}_{start}",
                        use_container_width=True,
                    ):
                        toggle_saved(book)


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================
with st.sidebar:
    st.html(
        """
        <div style="padding:0.5rem 0 1.5rem 0">
            <div class="brand-mark">📚 PageTurnner</div>
            <div class="brand-subtitle">A little more to read</div>
        </div>
        """,
    )

    st.markdown("<p style='font-size:0.72rem;letter-spacing:0.15em;font-weight:800;color:#a9c0ae'>YOUR SPACE</p>", unsafe_allow_html=True)
    page = st.radio(
        "Navigate",
        ["Discover", "Popular books", "My reading shelf", "About PageTurnner"],
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown(f"**{len(books):,} books** in your library")
    st.markdown(f"**{len(st.session_state.reading_list)} saved** on your shelf")
    st.divider()
    st.caption("Built with Python · Streamlit · Scikit-learn")


# ============================================================
# TOP BRAND STRIP
# ============================================================
left, right = st.columns([5, 1])
with left:
    st.markdown("<div class='section-kicker'>THE ART OF FINDING A GOOD BOOK</div>", unsafe_allow_html=True)
with right:
    st.markdown(
        f"<div style='text-align:right;color:#6d786f;font-size:0.85rem;padding-top:0.2rem'>♥ {len(st.session_state.reading_list)} saved</div>",
        unsafe_allow_html=True,
    )


# ============================================================
# DISCOVER PAGE
# ============================================================
if page == "Discover":
    st.html(
        """
        <section class="hero">
            <div class="hero-kicker">YOUR PERSONAL READING COMPANION</div>
            <div class="hero-title">Find your next<br>favourite story.</div>
            <div class="hero-copy">Tell us a book you love. PageTurnner will look for your next great read using the themes, authors and tags that connect books.</div>
        </section>
        """
    )

    metric1, metric2, metric3 = st.columns(3)
    metric1.metric("Books to explore", f"{len(books):,}")
    metric2.metric("Recommendation engine", "TF-IDF")
    metric3.metric("Your saved shelf", len(st.session_state.reading_list))

    st.write("")
    st.markdown("### Find books that feel like your kind of story")
    st.caption("Search a title, choose a few preferences, and discover similar reads.")

    surprise_col, hint_col = st.columns([1, 3])
    with surprise_col:
        if st.button("🎲 Surprise me", use_container_width=True):
            pool = books.copy()
            if "ratings_count" in pool.columns:
                pool = pool.sort_values("ratings_count", ascending=False).head(300)
            pool = pool[pool["title"].str.strip() != ""]
            if not pool.empty:
                surprise_title = random.choice(pool["title"].tolist())
                st.session_state["book_search"] = surprise_title
                st.rerun()
    with hint_col:
        st.markdown("<div class='small-note' style='padding:0.6rem 0'>Not sure what to search? Let PageTurnner pick a starting point.</div>", unsafe_allow_html=True)

    search_query = st.text_input(
        "Search by book title",
        placeholder="Try The Hunger Games, Harry Potter, Pride and Prejudice...",
        key="book_search",
    )

    if search_query.strip():
        matches = books[books["title"].str.contains(search_query.strip(), case=False, na=False, regex=False)]
        title_options = matches["title"].drop_duplicates().head(60).tolist()
    else:
        title_options = []

    if title_options:
        selected_book = st.selectbox("Choose a book", title_options, key="selected_book")

        f1, f2, f3 = st.columns([1.15, 1, 1])
        with f1:
            selected_genre = st.selectbox("Explore a genre", GENRE_OPTIONS, key="recommend_genre")
        with f2:
            minimum_rating = st.slider("Minimum rating", 0.0, 5.0, 0.0, 0.5, key="recommend_rating")
        with f3:
            sort_option = st.selectbox("Sort results", ["Best match", "Highest rated"], key="recommend_sort")

        if st.button("✨ Find my next read", type="primary", use_container_width=True):
            try:
                with st.spinner("Finding books with similar themes and reading tags..."):
                    candidates = recommend_books(selected_book, books, tfidf_matrix, top_n=80)
                    st.session_state["recommendation_candidates"] = candidates
                    st.session_state["recommendation_source"] = selected_book
            except Exception as error:
                st.error(f"Could not create recommendations: {error}")

        if st.session_state.get("recommendation_source") == selected_book and "recommendation_candidates" in st.session_state:
            candidates = st.session_state["recommendation_candidates"].copy()
            candidates = filter_by_genre(candidates, selected_genre)

            if "average_rating" in candidates.columns:
                candidates = candidates[candidates["average_rating"].fillna(0) >= minimum_rating]

            if sort_option == "Highest rated" and "average_rating" in candidates.columns:
                candidates = candidates.sort_values("average_rating", ascending=False)
            elif "similarity_score" in candidates.columns:
                candidates = candidates.sort_values("similarity_score", ascending=False)

            display_count = 8
            candidates = candidates.head(display_count)
            st.divider()
            st.markdown("### ✨ Your next reads")
            st.caption(f"Books related to **{selected_book}** · ranked using content similarity")
            render_book_grid(candidates, key_prefix="recommend", show_similarity=True)

    elif search_query.strip():
        st.warning("No matching titles found. Try a shorter or different search.")
    else:
        st.markdown("### A few reader favourites to get you started")
        if "ratings_count" in books.columns:
            featured = books.sort_values("ratings_count", ascending=False).drop_duplicates("title").head(4)
        else:
            featured = books.drop_duplicates("title").head(4)
        render_book_grid(featured, key_prefix="featured", show_similarity=False)


# ============================================================
# POPULAR BOOKS PAGE
# ============================================================
elif page == "Popular books":
    st.html(
        """
        <section class="hero">
            <div class="hero-kicker">READER FAVOURITES</div>
            <div class="hero-title">Books people<br>keep coming back to.</div>
            <div class="hero-copy">Browse well-known titles from the collection and save the ones you want to read.</div>
        </section>
        """
    )

    p1, p2 = st.columns(2)
    with p1:
        popular_genre = st.selectbox("Genre", GENRE_OPTIONS, key="popular_genre")
    with p2:
        popular_min_rating = st.slider("Minimum rating", 0.0, 5.0, 0.0, 0.5, key="popular_rating")

    popular = books.copy()
    popular = filter_by_genre(popular, popular_genre)
    if "average_rating" in popular.columns:
        popular = popular[popular["average_rating"].fillna(0) >= popular_min_rating]
    if "ratings_count" in popular.columns:
        popular = popular.sort_values("ratings_count", ascending=False)
    popular = popular.drop_duplicates("title").head(20)
    st.markdown(f"### Popular picks <span style='color:#8a948b;font-size:0.9rem'>({len(popular)} shown)</span>", unsafe_allow_html=True)
    render_book_grid(popular, key_prefix="popular", show_similarity=False)


# ============================================================
# READING SHELF PAGE
# ============================================================
elif page == "My reading shelf":
    st.html(
        """
        <section class="hero">
            <div class="hero-kicker">YOUR PERSONAL BOOKSHELF</div>
            <div class="hero-title">Keep the stories<br>you want to remember.</div>
            <div class="hero-copy">Save interesting books here. Your shelf is stored locally in this project, so it remains available after you close the app.</div>
        </section>
        """
    )

    saved_ids = set(st.session_state.reading_list)
    if not saved_ids:
        st.info("Your shelf is empty. Visit Discover or Popular books and choose 'Save to shelf' on a book you like.")
    else:
        shelf = books[books["book_id"].astype(str).isin(saved_ids)].copy() if "book_id" in books.columns else books.iloc[0:0].copy()
        st.markdown(f"### Your saved books · {len(shelf)}")
        render_book_grid(shelf, key_prefix="shelf", show_similarity=False)
        if st.button("Clear my reading shelf", type="secondary"):
            st.session_state.reading_list = []
            save_reading_list([])
            st.rerun()


# ============================================================
# ABOUT PAGE
# ============================================================
elif page == "About PageTurnner":
    st.html(
        """
        <section class="hero">
            <div class="hero-kicker">ABOUT THE PROJECT</div>
            <div class="hero-title">A smarter way<br>to discover stories.</div>
            <div class="hero-copy">PageTurnner is a book recommendation project that uses book information and reader-generated tags to find related titles.</div>
        </section>
        """
    )

    st.markdown("### How PageTurnner recommends books")
    a, b = st.columns(2)
    with a:
        with st.container(border=True):
            st.markdown("#### 01 · Understand the book")
            st.write("The system combines titles, authors and tags into a text profile for each book.")
        with st.container(border=True):
            st.markdown("#### 02 · Turn words into features")
            st.write("TF-IDF converts important words and phrases into numerical vectors.")
    with b:
        with st.container(border=True):
            st.markdown("#### 03 · Compare book profiles")
            st.write("Cosine similarity compares the selected book with the other book profiles.")
        with st.container(border=True):
            st.markdown("#### 04 · Show related reads")
            st.write("Books with stronger content similarity appear higher in the recommendations.")

    st.markdown("### Technology stack")
    st.write("Python · Pandas · NumPy · Scikit-learn · TF-IDF · Cosine Similarity · Streamlit")
    st.info("This version uses content-based filtering. It does not yet learn an individual reader's preferences from a personal rating history.")


# ============================================================
# FOOTER
# ============================================================
st.divider()
st.markdown(
    "<div class='footer'>📚 <b>PageTurnner</b> &nbsp;·&nbsp; Find a story. Make it yours. &nbsp;·&nbsp; Built with Python and Machine Learning</div>",
    unsafe_allow_html=True,
)
