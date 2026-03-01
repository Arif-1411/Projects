from flask import Flask, render_template, request
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import joblib
import os
import urllib.request

app = Flask(__name__)

DATA_URL = "https://raw.githubusercontent.com/nicholasjhana/short-term-energy-demand-forecasting/master/data/retail_transaction_dataset.csv"

# Fallback local dataset (if URL fails)
FALLBACK_DATA = [
    ['milk', 'bread', 'butter'],
    ['bread', 'butter', 'jam'],
    ['milk', 'bread', 'jam'],
    ['milk', 'butter'],
    ['bread', 'jam', 'butter'],
    ['milk', 'bread'],
    ['butter', 'jam'],
    ['milk', 'bread', 'butter', 'jam'],
    ['bread', 'butter'],
    ['milk', 'jam'],
    ['bread', 'milk', 'butter'],
    ['jam', 'bread'],
    ['milk', 'bread', 'jam', 'butter'],
    ['butter', 'bread'],
    ['milk', 'jam', 'bread'],
    ['bread', 'butter', 'milk'],
    ['jam', 'milk'],
    ['bread', 'jam'],
    ['milk', 'butter', 'jam'],
    ['bread', 'milk'],
]

rules_df = None


def load_and_clean_data():
    print("📥 Loading dataset...")

    try:
        # Try loading from URL using requests style
        df = pd.read_csv(DATA_URL)
        print("✅ Downloaded dataset. Columns:", df.columns.tolist())

        # This dataset has columns like: BillNo, Itemname, Quantity, etc.
        # We need to group by BillNo → list of items
        transactions = df.groupby('BillNo')['Itemname'].apply(list).tolist()
        print("Total transactions:", len(transactions))

    except Exception as e:
        print(f"⚠️ URL failed: {e}")
        print("📦 Using fallback dataset...")
        transactions = FALLBACK_DATA

    return transactions


def convert_to_binary(transactions):
    print("🧹 Encoding transactions...")
    te = TransactionEncoder()
    te_array = te.fit_transform(transactions)
    df_encoded = pd.DataFrame(te_array, columns=te.columns_)
    print("Shape:", df_encoded.shape)
    return df_encoded


def train_apriori():
    print("🤖 Training Apriori...")

    transactions = load_and_clean_data()
    df_encoded = convert_to_binary(transactions)

    frequent_items = apriori(
        df_encoded,
        min_support=0.01,
        use_colnames=True
    )
    print("Frequent itemsets found:", len(frequent_items))

    rules = association_rules(
        frequent_items,
        metric="lift",
        min_threshold=1.0,
        num_itemsets=len(frequent_items)
    )
    print("Rules found:", len(rules))

    rules = rules.sort_values(by="lift", ascending=False)

    global rules_df
    rules_df = rules

    joblib.dump(rules, "apriori_rules.joblib")
    print("✅ Model saved!")

    # Print top 5 rules
    print("\nTop 5 rules:")
    for _, row in rules.head(5).iterrows():
        print(f"  {set(row['antecedents'])} => {set(row['consequents'])} | lift={row['lift']:.2f}")

    return rules


def load_or_train():
    global rules_df
    if os.path.exists("apriori_rules.joblib"):
        print("📂 Loading saved rules...")
        rules_df = joblib.load("apriori_rules.joblib")
        print("Rules loaded:", len(rules_df))
    else:
        rules_df = train_apriori()


def recommend_items(item_name):
    item_name = item_name.strip()
    print("🔍 Finding recommendations for:", item_name)

    # Case-insensitive match
    matches = rules_df[rules_df['antecedents'].apply(
        lambda x: item_name.lower() in [i.lower() for i in list(x)]
    )]

    if matches.empty:
        # Try partial match
        matches = rules_df[rules_df['antecedents'].apply(
            lambda x: any(item_name.lower() in i.lower() for i in list(x))
        )]

    if matches.empty:
        return []

    matches = matches.sort_values(by="lift", ascending=False)

    recommendations = []
    for _, row in matches.iterrows():
        for item in list(row['consequents']):
            if item.lower() != item_name.lower():
                recommendations.append({
                    'item': item,
                    'support': round(float(row['support']), 4),
                    'confidence': round(float(row['confidence']), 4),
                    'lift': round(float(row['lift']), 4)
                })

    # Deduplicate by item name
    seen = set()
    unique_recs = []
    for r in recommendations:
        if r['item'] not in seen:
            seen.add(r['item'])
            unique_recs.append(r)

    return unique_recs[:5]


def get_all_items():
    all_items = set()
    for items in rules_df['antecedents']:
        all_items.update(items)
    for items in rules_df['consequents']:
        all_items.update(items)
    return sorted(all_items)


@app.route('/')
def home():
    all_items = get_all_items()
    total_rules = len(rules_df)
    return render_template(
        "index.html",
        all_items=all_items,
        total_rules=total_rules
    )


@app.route('/train', methods=['POST'])
def retrain():
    train_apriori()
    return "<h1>✅ Apriori Model Retrained</h1><a href='/'>Back</a>"


@app.route('/predict', methods=['POST'])
def predict():
    item_input = request.form.get("item", "").strip()
    recs = recommend_items(item_input)

    return render_template(
        "result.html",
        item=item_input,
        recs=recs
    )


if __name__ == "__main__":
    load_or_train()
    app.run(debug=True)