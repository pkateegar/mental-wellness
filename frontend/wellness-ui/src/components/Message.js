import React from "react";

const Message = ({ text, sender, sentiment }) => {
    const getSentimentColor = (sentiment) => {
        switch (sentiment) {
            case "positive":
                return "green";
            case "negative":
                return "red";
            case "neutral":
                return "gray";
            default:
                return "black";
        }
    };

    return (
        <div className={`message ${sender}`}>
            <p style={{ color: getSentimentColor(sentiment) }}>{text}</p>
        </div>
    );
};

export default Message;