import re
from modules.post_process import extract_json_blocks, parse_string


def post_process_drug(prob, response, output):
    overview = response.split("Overview")[-1].split("Knowledge")[0]
    knowledge = response.split("Knowledge")[-1].split("Comparison")[0]
    comparison = response.split("Comparison")[-1].split("Reasoning")[0]
    reasoning = response.split("Reasoning")[-1].split("Conclusion")[0]
    conclusion = response.split("Conclusion")[-1].split("Confidence")[0]
    confidence = response.split("Confidence")[-1]

    assert overview != knowledge and knowledge != comparison and comparison != reasoning and reasoning != conclusion

    # remove grounds truth in the reasoning
    reasoning = re.sub(f"score:? \d", "", reasoning)
    reasoning = re.sub(f"Score:? \d", "", reasoning)

    confidence = [w for w in ["Very Confident", "Very Not Confident", "Not Confident", "Confident", "Neutral"] if w in confidence][0]

    prediction = {}
    for key, value in extract_json_blocks(conclusion).items():
        assert isinstance(value, dict)
        assert 'reason' in value and 'score' in value
        value['score'] = int(value['score'])
        prediction[key] = dict(value)

    result = {"response": response,
              "overview": parse_string(overview),
              "knowledge": parse_string(knowledge),
              "comparison": parse_string(comparison),
              "reasoning": parse_string(reasoning),
              "prediction": prediction,
              "confidence": confidence}
    return prob, result, output


POST_PROCESS_FN = {"mortality": None, "readmission": None,
                   "lenofstay": None, "drugrec": post_process_drug,
                   "diagnosis": post_process_drug}
