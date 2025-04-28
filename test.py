import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

# --- PAGE CONFIG ---
st.set_page_config(page_title="Titanic Survival Analysis", page_icon="🚢")
st.title('🚢 Titanic Survival Analysis')

# --- LOAD DATA ---
def load_csv():
    df = pd.read_csv("titanic.csv")
    df['Age'] = df['Age'].fillna(0)
    df['Cabin'] = df['Cabin'].fillna(0)
    df['Fare'] = df['Fare'].fillna(0)
    return df

df = load_csv()

# --- SIDEBAR FILTERS ---
st.sidebar.header("🧰 Filter Options")

pclass_filter = st.sidebar.multiselect("Passenger Class", options=sorted(df["Pclass"].unique()), default=sorted(df["Pclass"].unique()))
sex_filter = st.sidebar.multiselect("Sex", options=df["Sex"].unique(), default=df["Sex"].unique())
min_age, max_age = int(df["Age"].min()), int(df["Age"].max())
age_range = st.sidebar.slider("Age Range", min_age, max_age, (min_age, max_age))

# --- FILTER DATA ---
filtered_df = df[
    (df["Pclass"].isin(pclass_filter)) &
    (df["Sex"].isin(sex_filter)) &
    (df["Age"].between(age_range[0], age_range[1], inclusive="both"))
]

# --- SEARCH PASSENGER ---
search = st.text_input("🔎 Search for a passenger by name")
search_df = filtered_df.copy()
if search:
    search_df = search_df[search_df["Name"].str.contains(search, case=False, na=False)]

    if not search_df.empty:
        for _, row in search_df.iterrows():
            with st.expander(f"👤 {row['Name']}"):
                st.markdown(f"**Age:** {row['Age']}")
                st.markdown(f"**Sex:** {row['Sex']}")
                st.markdown(f"**Class:** {row['Pclass']}")
                st.markdown(f"**Fare Paid:** ${row['Fare']:.2f}")
                st.markdown(f"**Embarked From:** {row['Embarked']}")
                st.markdown(f"**Survived:** {'✅ Yes' if row['Survived'] == 1 else '❌ No'}")
    else:
        st.warning("No passenger found with that name.")

#  Survival Rate by Passenger Class
st.subheader("📊 Survival Rate by Passenger Class")
total = filtered_df.groupby('Pclass')['Survived'].count()
survived = filtered_df.groupby('Pclass')['Survived'].sum()
survival_rate = survived / total
sdf = pd.DataFrame({'Pclass': survival_rate.index, 'Survival_Rate': survival_rate.values})
fig = px.bar(sdf,x="Pclass",y="Survival_Rate",color="Pclass", title="Survival Rate by Passenger Class",labels={"Survival_Rate": "Survival Rate", "Pclass": "Passenger Class"},text_auto='.2f')
st.plotly_chart(fig, use_container_width=True)

# survivors by sex Pie Chart
st.subheader("🧑‍🤝‍🧑 Total Survivors by Sex")
survive_sex = df.groupby('Sex')['Survived'].sum().reset_index()
fig2 = px.pie(survive_sex,names="Sex",values="Survived",title="Survivors Distribution by Sex", color_discrete_sequence=px.colors.sequential.RdBu)
st.plotly_chart(fig2, use_container_width=True)

#Average Age by Class
st.subheader("📈 Average Age by Class")
aage = filtered_df.groupby('Pclass')['Age'].mean()
age_df = pd.DataFrame({'Class': aage.index, 'Average Age': aage.values})
fig3, ax3 = plt.subplots()
sns.barplot(x='Class', y='Average Age', data=age_df, ax=ax3)
st.pyplot(fig3)

#Average Fare by Class
st.subheader("💰 Average Fare by Class")
fare = filtered_df.groupby('Pclass')['Fare'].mean()
fare_df = pd.DataFrame({'Class': fare.index, 'Average Fare': fare.values})
fig4, ax4 = plt.subplots()
sns.barplot(x='Class', y='Average Fare', data=fare_df, ax=ax4)
st.pyplot(fig4)

#Survivors by Embarkation Point
st.subheader("🛳️ Survivors by Embarkation Point")
e = filtered_df.groupby('Embarked')['Survived'].sum()
embark_df = pd.DataFrame({'Embarkation Point': e.index, 'Number of Survivors': e.values})
fig5, ax5 = plt.subplots()
sns.barplot(x='Embarkation Point', y='Number of Survivors', data=embark_df, ax=ax5)
st.pyplot(fig5)

#Survival Rate by Class and Sex (Heatmap)
st.subheader("🔥 Survival Rate by Class and Sex (Heatmap)")
heatmap_data = filtered_df.pivot_table(values="Survived", index="Sex", columns="Pclass", aggfunc="mean")
fig, ax = plt.subplots()
sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="YlGnBu", ax=ax)
st.pyplot(fig)

#Passenger Distribution by Port of Embarkation
st.subheader("🥧 Passenger Distribution by Port of Embarkation")
embarked_counts = filtered_df["Embarked"].value_counts()
fig, ax = plt.subplots()
ax.pie(embarked_counts, labels=embarked_counts.index, autopct="%1.1f%%", startangle=90)
ax.set_title("Passenger Distribution by Port of Embarkation")
plt.axis("equal")
fig.tight_layout()
st.pyplot(fig)

#the raw dataset
st.subheader("🗃️ Raw Titanic Data")
st.dataframe(filtered_df, use_container_width=True)

#about
with st.expander("ℹ️ About"):
    st.write("This app shows survival statistics from the Titanic dataset including survival rates, age and fare averages as an assigment ")
