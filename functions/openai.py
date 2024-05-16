# Imports
import os
import ast
import sys
import json
import openai
from time import sleep
from dotenv import load_dotenv


# Usa a função de chat da OpenAI para extrair informações
def _chat_completion_chatgpt(prompt: str, tool: list[dict], tool_choice: str | dict, temperature: float, model: str) -> dict:
    max_retries = 3
    errors = []
    for t in range(max_retries):
        try:
            sys.path.append('..')
            load_dotenv()
            openai.api_key = os.environ.get('OPEN_API_KEY')
            response = openai.ChatCompletion.create(
                model=os.environ.get('CHAT_MODEL'),
                messages=[
                    {'role': 'system', 'content': 'You are an obedient assistant who only responds to what is requested'},
                    {'role': 'user', 'content': prompt}
                ],
                temperature=temperature,
                tools=tool,
                tool_choice=tool_choice,
            )
        except (openai.APIError,
                openai.InvalidRequestError,
                openai.OpenAIError) as error:
            errors.append(error)
            print(f'Falha ao obter resposta de \'{prompt[:30]}...\', tentando novamente... '
                  f'({t+1}/{max_retries}): {error}', file=sys.stderr)
            sleep(5)
        else:
            try:
                # Parece que existe um bug no OpenAI que às vezes gera um json inválido com vírgula no último elemento
                return json.loads(
                    json.dumps(ast.literal_eval(response.choices[0].message.tool_calls[0].function.arguments))
                )
            except SyntaxError as error:
                print(f'Falha ao obter resposta de \'{prompt[:30]}...\' tentando novamente... '
                      f'({t+1}/{max_retries}): {error}', file=sys.stderr)
    else:
        raise Exception(f'Falha ao obter resposta de \'{prompt[:30]}...\': {errors}')
    
    
# Função para gerar temas e subtemas a partir de um pedaço de texto
def gerar_informacoes(information: str) -> dict | None:
    tool = [
        {
            'type': 'function',
            'function': {
                'name': 'sent_info',
                'description': 'Retrieve sentiment and relevant information from text.',
                'parameters': {
                    'type': 'object',
                    'description': 'JSON object with the sentiment and relevant information from text',
                    'properties': {
                        'sentiment': {
                            'type': 'string',
                            'description': 'The sentiment of the text',
                            'items': {'type': 'string'},
                            'maxItems': 1,
                            'minItems': 1,
                            'uniqueItems': True,
                        },
                        'info': {
                            'type': 'array',
                            'description': 'List of relevant information',
                            'items': {'type': 'string'},
                            'maxItems': 20,
                            'minItems': 10,
                            'uniqueItems': True,
                        }
                    }
                }
            }
        }
    ]
    
    example = {'sentiment': 'positive', 'info': ['info_1',...]}
    
    prompt = f'''Consider the text between <<>> and follow the instructions below one by one:
                 - Assign the sentiment of the text between <<>> as 'positive', 'negative' or 'neutral'
                 - Extract at least 10 relevant infomrations or entities from the text between <<>>
                 - The output should always be a JSON object with the sentiment and the info and nothing else, like this example: {example}
                 <<{information}>>'''
    
    obj = _chat_completion_chatgpt(
                prompt=prompt,
                model=os.environ.get('CHAT_MODEL'),
                tool=tool,
                tool_choice={'type': 'function', 'function': {'name': 'sent_info'}},
                temperature=0)
    
    return obj