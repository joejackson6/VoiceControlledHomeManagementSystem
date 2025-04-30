import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
import joblib
import numpy as np

class IntentMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(IntentMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

def get_training_data():
    return [
        # TIME
        ("what time is it", "get_time"),
        ("tell me the time", "get_time"),
        ("do you know the time", "get_time"),
        ("can you tell the time", "get_time"),
        ("current time please", "get_time"),
        ("give me the time", "get_time"),
        ("what's the time now", "get_time"),
        ("time", "get_time"),
        ("what time is it right now", "get_time"),

        # WEATHER
        ("how's the weather", "get_weather"),
        ("what's the forecast", "get_weather"),
        ("is it going to rain", "get_weather"),
        ("tell me the weather", "get_weather"),
        ("weather report", "get_weather"),
        ("what's the weather like", "get_weather"),
        ("will it be sunny", "get_weather"),
        ("check the weather", "get_weather"),
        ("is it hot outside", "get_weather"),
        ("weather right now", "get_weather"),
        ("what's it like outside", "get_weather"),
        ("is it cold today", "get_weather"),

        # LIGHT CONTROL
        ("turn on the light", "turn_on_light"),
        ("switch on the bulb", "turn_on_light"),
        ("light on", "turn_on_light"),
        ("can you turn the light on", "turn_on_light"),
        ("power on the lamp", "turn_on_light"),
        ("activate the light", "turn_on_light"),
        ("illuminate the room", "turn_on_light"),
        ("enable the light", "turn_on_light"),

        ("turn off the light", "turn_off_light"),
        ("switch off the bulb", "turn_off_light"),
        ("light off", "turn_off_light"),
        ("can you turn the light off", "turn_off_light"),
        ("power down the lamp", "turn_off_light"),
        ("disable the light", "turn_off_light"),
        ("kill the lights", "turn_off_light"),

        # GREETINGS
        ("hello", "greeting"),
        ("hi", "greeting"),
        ("hey", "greeting"),
        ("howdy", "greeting"),
        ("yo", "greeting"),
        ("good morning", "greeting"),
        ("good afternoon", "greeting"),
        ("good evening", "greeting"),
        ("greetings", "greeting"),
        ("sup", "greeting"),
        ("what's up", "greeting"),
        ("hiya", "greeting"),

        # IDENTITY
        ("what's your name", "identity"),
        ("who are you", "identity"),
        ("tell me your name", "identity"),
        ("do you have a name", "identity"),
        ("what should I call you", "identity"),
        ("who made you", "identity"),
        ("who created you", "identity"),
        ("who built you", "identity"),
        ("who programmed you", "identity"),

        # THANKS / GOODBYE
        ("thanks", "thanks"),
        ("thank you", "thanks"),
        ("i appreciate it", "thanks"),
        ("much appreciated", "thanks"),

        ("goodbye", "goodbye"),
        ("bye", "goodbye"),
        ("see you later", "goodbye"),
        ("talk to you soon", "goodbye"),
        ("peace out", "goodbye"),

        # UNKNOWN – expanded significantly
        ("can you play music", "unknown"),
        ("play something", "unknown"),
        ("what's the news", "unknown"),
        ("tell me a joke", "unknown"),
        ("what's 5 plus 7", "unknown"),
        ("can you do math", "unknown"),
        ("set a timer for 5 minutes", "unknown"),
        ("remind me to call mom", "unknown"),
        ("can you send a text", "unknown"),
        ("navigate to walmart", "unknown"),
        ("how do I get to the airport", "unknown"),
        ("what's your favorite movie", "unknown"),
        ("can you dance", "unknown"),
        ("do you like pizza", "unknown"),
        ("what do you think of me", "unknown"),
        ("are you alive", "unknown"),
        ("do you sleep", "unknown"),
        ("do you believe in god", "unknown"),
        ("who is the president", "unknown"),
        ("what's the capital of france", "unknown"),
        ("read me a bedtime story", "unknown"),
        ("translate hello to spanish", "unknown"),
        ("do aliens exist", "unknown"),
        ("flip a coin", "unknown"),
        ("roll a dice", "unknown"),
        ("what's 100 divided by 7", "unknown"),
        ("how far is the moon", "unknown"),
        ("open youtube", "unknown"),
        ("can you call dad", "unknown"),
        ("show me pictures of cats", "unknown"),
        ("send an email", "unknown"),
        ("order me a pizza", "unknown")
    ]

def main():
    data = get_training_data()
    texts, labels = zip(*data)

    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 5))  # char-level n-grams
    X = vectorizer.fit_transform(texts).toarray()

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    input_dim = X_train.shape[1]
    hidden_dim = 128
    output_dim = len(set(y))

    model = IntentMLP(input_dim, hidden_dim, output_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    for epoch in range(200):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
        if epoch % 20 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    # Evaluate
    model.eval()
    with torch.no_grad():
        preds = model(X_test_tensor).argmax(dim=1).numpy()
        acc = accuracy_score(y_test, preds)
        print(f"Validation Accuracy: {acc:.2f}")

    torch.save(model.state_dict(), "intent_char_mlp.pt")
    joblib.dump(vectorizer, "intent_vectorizer.pkl")
    joblib.dump(label_encoder, "intent_label_encoder.pkl")
    print("Model and encoders saved.")

if __name__ == "__main__":
    main()
