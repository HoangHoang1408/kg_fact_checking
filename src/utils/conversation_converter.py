import re

def convert_formatted_text_to_json(formatted_text):
    # Split the text into lines and remove empty lines
    lines = [line.strip() for line in formatted_text.split('\n') if line.strip()]
    
    # Skip the header line if present
    start_idx = 0
    if '### Translated Conversation ###' in lines[0]:
        start_idx = 1
    
    messages = []
    for line in lines[start_idx:]:
        # Extract the role and content using regex
        match = re.match(r'<(USER|ASSISTANT)>:\s*(.*)', line, re.IGNORECASE)
        if match:
            role, content = match.groups()
            # Normalize role to lowercase
            role = role.lower()
            messages.append({
                "role": role,
                "content": content.strip()
            })
    
    return {"messages": messages}

# Example usage:
if __name__ == "__main__":
    example_text = '''
    ### Translated Conversation ###

    <USER>: user turn

    <ASSISTANT>: assistant turn

    <User>: another user turn
    <ASSISTANT>: another assistant turn
    '''
    
    result = convert_formatted_text_to_json(example_text)
    print(result)
