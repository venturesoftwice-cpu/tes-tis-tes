// index.js - OpenAI to NVIDIA NIM & Google AI Studio API Proxy
const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();

app.use(cors());
// Set incoming payload size limit to 4.5 MB to allow up to 1M token context windows
app.use(express.json({ limit: '4.5mb' }));
app.use(express.urlencoded({ limit: '4.5mb', extended: true }));

const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY = process.env.NIM_API_KEY;
const GEMINI_API_KEY = process.env.GEMINI_API_KEY;

const SHOW_REASONING = true; 
const ENABLE_THINKING_MODE = true; 

const MODEL_MAPPING = {
  'gpt-3.5-turbo': 'thinkingmachines/inkling',
  'gpt-4': 'qwen/qwen3-coder-480b-a35b-instruct',
  'gpt-4-turbo': 'google/gemma-4-31b-it',
  'gpt-4o': 'deepseek-ai/deepseek-v4-pro', 
  'claude-3-opus': 'z-ai/glm-5.2',
  'claude-3-sonnet': 'deepseek-ai/deepseek-v4-flash',
  'gemini-pro': 'google/diffusiongemma-26b-a4b-it',
  
  // Google AI Studio Models
  'gemini-3-flash': 'gemini-3.6-flash',
  'gemma-4-31b': 'gemma-4-31b-it',
  'gemma-4-26b': 'gemma-4-26b-a4b-it',
  
  // NIM Model
  'mistral-medium-3.5': 'mistralai/mistral-medium-3.5-128b'
};

// Helper function to dynamically adjust parameters based on the specific NIM model
function getModelConfig(nimModel, enableThinking) {
  let maxTokens = 16384; 
  let chatTemplateKwargs = undefined;

  const modelLower = nimModel.toLowerCase();

  if (modelLower.includes('gemma-4')) {
    maxTokens = 16384;
    chatTemplateKwargs = enableThinking ? { enable_thinking: true } : undefined;
  } 
  else if (modelLower.includes('deepseek-v4') || modelLower.includes('deepseek')) {
    maxTokens = 16384;
    chatTemplateKwargs = enableThinking ? { thinking: true, reasoning_effort: "high" } : undefined;
  }
  else if (modelLower.includes('inkling')) {
    maxTokens = 8192; // Inkling's max output token limit
    chatTemplateKwargs = undefined; 
  }
  else if (modelLower.includes('glm-5.2')) {
    maxTokens = 16384;
    chatTemplateKwargs = enableThinking ? { thinking: { type: "enabled" }, reasoning_effort: "high" } : undefined;
  }
  else if (modelLower.includes('diffusiongemma') || modelLower.includes('gemma-26b')) {
    maxTokens = 4096;
    chatTemplateKwargs = enableThinking ? { enable_thinking: true } : undefined;
  }
  else if (modelLower.includes('mistral-medium-3.5')) {
    maxTokens = 16384;
    chatTemplateKwargs = undefined;
  }

  return { maxTokens, chatTemplateKwargs };
}

app.get('/health', (req, res) => {
  res.json({ 
    status: 'ok', 
    service: 'OpenAI to NVIDIA NIM Proxy', 
    reasoning_display: SHOW_REASONING,
    thinking_mode: ENABLE_THINKING_MODE
  });
});

app.get('/v1/models', (req, res) => {
  const models = Object.keys(MODEL_MAPPING).map(model => ({
    id: model,
    object: 'model',
    created: Date.now(),
    owned_by: 'nvidia-nim-proxy'
  }));
  res.json({ object: 'list', data: models });
});

app.post('/v1/chat/completions', async (req, res) => {
  try {
    const { model, messages, temperature, max_tokens, stream } = req.body;
    
    // Auth fallback configurations
    const authHeader = req.headers.authorization;
    const clientApiKey = authHeader ? authHeader.replace('Bearer ', '').trim() : '';
    const activeApiKey = NIM_API_KEY || clientApiKey;

    let nimModel = MODEL_MAPPING[model];
    if (!nimModel) {
      try {
        await axios.post(`${NIM_API_BASE}/chat/completions`, {
          model: model,
          messages: [{ role: 'user', content: 'test' }],
          max_tokens: 1
        }, {
          headers: { 'Authorization': `Bearer ${activeApiKey}`, 'Content-Type': 'application/json' },
          validateStatus: (status) => status < 500
        }).then(res => {
          if (res.status >= 200 && res.status < 300) {
            nimModel = model;
          }
        });
      } catch (e) {}
      
      if (!nimModel) {
        nimModel = 'deepseek-ai/deepseek-v4-pro';
      }
    }

    // Determine destination pathway
    const isAIStudio = nimModel && (nimModel.startsWith('gemini-') || nimModel.startsWith('gemma-4'));

    // --- Google AI Studio Pathway ---
    if (isAIStudio) {
      if (!GEMINI_API_KEY) {
        return res.status(401).json({
          error: { 
            message: 'API Key Gemini / Google AI Studio kosong. Silakan atur GEMINI_API_KEY di dashboard lingkungan Vercel Anda.', 
            type: 'invalid_request_error', 
            code: 401 
          }
        });
      }

      const isGemmaModel = nimModel.startsWith('gemma-4');

      // --- PATHWAY A: Gemma Standard Generation ---
      if (isGemmaModel) {
        const systemMessage = messages.find(m => m.role === 'system');
        const systemInstructionText = systemMessage ? systemMessage.content : '';

        const googleContents = messages.map(m => {
          if (m.role === 'system') return null;
          return {
            role: m.role === 'assistant' ? 'model' : 'user',
            parts: [{ text: m.content }]
          };
        }).filter(Boolean);

        const googleUrl = `https://generativelanguage.googleapis.com/v1beta/models/${nimModel}:${stream ? 'streamGenerateContent' : 'generateContent'}?key=${GEMINI_API_KEY}${stream ? '&alt=sse' : ''}`;

        const payload = {
          contents: googleContents,
          generationConfig: {
            thinkingConfig: {
              thinkingLevel: "high"
            }
          }
        };

        if (systemInstructionText) {
          payload.systemInstruction = {
            parts: [{ text: systemInstructionText }]
          };
        }

        const response = await axios.post(googleUrl, payload, {
          responseType: stream ? 'stream' : 'json'
        });

        if (stream) {
          res.setHeader('Content-Type', 'text/event-stream');
          res.setHeader('Cache-Control', 'no-cache');
          res.setHeader('Connection', 'keep-alive');

          let buffer = '';
          let thinkingOpened = false;

          response.data.on('data', (chunk) => {
            buffer += chunk.toString();
            const lines = buffer.split('\n');
            buffer = lines.pop() || '';

            lines.forEach(line => {
              const trimmed = line.trim();
              if (!trimmed) return;

              if (trimmed.startsWith('data: ')) {
                const rawData = trimmed.slice(6);
                try {
                  const data = JSON.parse(rawData);
                  const parts = data.candidates?.[0]?.content?.parts;

                  if (parts && Array.isArray(parts)) {
                    let textToSend = "";
                    
                    parts.forEach(part => {
                      const text = part.text || "";
                      const isThought = part.thought === true;

                      if (isThought) {
                        if (!thinkingOpened) {
                          textToSend += '<think>\n';
                          thinkingOpened = true;
                        }
                        textToSend += text;
                      } else {
                        if (thinkingOpened && text.trim().length > 0) {
                          textToSend += '\n</think>\n\n';
                          thinkingOpened = false;
                        }
                        textToSend += text;
                      }
                    });

                    if (textToSend) {
                      const clientPayload = {
                        id: `chatcmpl-${Date.now()}`,
                        object: 'chat.completion.chunk',
                        created: Math.floor(Date.now() / 1000),
                        model: model,
                        choices: [{
                          index: 0,
                          delta: { content: textToSend },
                          finish_reason: null
                        }]
                      };
                      res.write(`data: ${JSON.stringify(clientPayload)}\n\n`);
                    }
                  }
                } catch (e) {
                  // Skip parsing metadata
                }
              }
            });
          });

          response.data.on('end', () => {
            if (thinkingOpened) {
              const closeThinkingPayload = {
                id: `chatcmpl-${Date.now()}`,
                object: 'chat.completion.chunk',
                created: Math.floor(Date.now() / 1000),
                model: model,
                choices: [{
                  index: 0,
                  delta: { content: '\n</think>\n\n' },
                  finish_reason: null
                }]
              };
              res.write(`data: ${JSON.stringify(closeThinkingPayload)}\n\n`);
            }
            res.write('data: [DONE]\n\n');
            res.end();
          });
        } else {
          let fullText = '';
          let thoughtsText = '';
          
          if (response.data.candidates?.[0]?.content?.parts) {
            response.data.candidates[0].content.parts.forEach(part => {
              if (part.thought === true && part.text) {
                thoughtsText += part.text;
              } else if (part.text) {
                fullText += part.text;
              }
            });
          }

          if (thoughtsText.trim()) {
            fullText = `<think>\n${thoughtsText.trim()}\n</think>\n\n` + fullText;
          }

          const openaiResponse = {
            id: `chatcmpl-${Date.now()}`,
            object: 'chat.completion',
            created: Math.floor(Date.now() / 1000),
            model: model,
            choices: [{
              index: 0,
              message: {
                role: 'assistant',
                content: fullText
              },
              finish_reason: 'stop'
            }],
            usage: {
              prompt_tokens: 0,
              completion_tokens: 0,
              total_tokens: 0
            }
          };
          res.json(openaiResponse);
        }
        return;
      }

      // --- PATHWAY B: Gemini Interactions API ---
      const formattedPrompt = messages.map(m => {
        const role = m.role === 'user' ? 'User' : 'Assistant';
        return `${role}: ${m.content}`;
      }).join('\n\n');

      const isBetaModel = nimModel.includes('preview') || nimModel.includes('3.6');
      const apiPath = isBetaModel ? 'v1beta' : 'v1';
      const googleUrl = `https://generativelanguage.googleapis.com/${apiPath}/interactions?key=${GEMINI_API_KEY}`;

      const response = await axios.post(
        googleUrl,
        {
          model: nimModel,
          input: formattedPrompt,
          generation_config: {
            thinking_summaries: "auto",
            thinking_level: "high"
          },
          stream: stream || false
        },
        {
          headers: {
            'Content-Type': 'application/json',
            'Api-Revision': '2026-05-20'
          },
          responseType: stream ? 'stream' : 'json'
        }
      );

      if (stream) {
        res.setHeader('Content-Type', 'text/event-stream');
        res.setHeader('Cache-Control', 'no-cache');
        res.setHeader('Connection', 'keep-alive');

        let buffer = '';
        let currentEvent = '';
        let thinkingOpened = false;

        response.data.on('data', (chunk) => {
          buffer += chunk.toString();
          const lines = buffer.split('\n');
          buffer = lines.pop() || '';

          lines.forEach(line => {
            const trimmed = line.trim();
            if (!trimmed) return;

            if (trimmed.startsWith('event: ')) {
              currentEvent = trimmed.slice(7);
            } else if (trimmed.startsWith('data: ')) {
              const rawData = trimmed.slice(6);
              if (rawData === '[DONE]') {
                res.write('data: [DONE]\n\n');
                return;
              }

              try {
                const data = JSON.parse(rawData);
                if (currentEvent === 'step.delta' || data.event_type === 'step.delta') {
                  const delta = data.delta;
                  if (delta) {
                    let textToSend = '';

                    // Intercept thought chunks
                    if (delta.type === 'thought_summary' && delta.content?.text) {
                      if (!thinkingOpened) {
                        textToSend += '<think>\n';
                        thinkingOpened = true;
                      }
                      textToSend += delta.content.text;
                    } 
                    // Intercept response text chunks
                    else if (delta.type === 'text' && delta.text) {
                      if (thinkingOpened) {
                        textToSend += '\n</think>\n\n';
                        thinkingOpened = false;
                      }
                      textToSend += delta.text;
                    }

                    if (textToSend) {
                      const clientPayload = {
                        id: `chatcmpl-${Date.now()}`,
                        object: 'chat.completion.chunk',
                        created: Math.floor(Date.now() / 1000),
                        model: model,
                        choices: [{
                          index: 0,
                          delta: { content: textToSend },
                          finish_reason: null
                        }]
                      };
                      res.write(`data: ${JSON.stringify(clientPayload)}\n\n`);
                    }
                  }
                }
              } catch (e) {
                // Skip parsing metadata
              }
            }
          });
        });

        response.data.on('end', () => {
          if (thinkingOpened) {
            const closeThinkingPayload = {
              id: `chatcmpl-${Date.now()}`,
              object: 'chat.completion.chunk',
              created: Math.floor(Date.now() / 1000),
              model: model,
              choices: [{
                index: 0,
                delta: { content: '\n</think>\n\n' },
                finish_reason: null
              }]
            };
            res.write(`data: ${JSON.stringify(closeThinkingPayload)}\n\n`);
          }
          res.write('data: [DONE]\n\n');
          res.end();
        });
      } else {
        let fullText = '';
        let thoughtsText = '';
        
        if (response.data.steps && Array.isArray(response.data.steps)) {
          response.data.steps.forEach(step => {
            if (step.type === 'thought' && step.summary) {
              step.summary.forEach(block => {
                if (block.type === 'text' && block.text) {
                  thoughtsText += block.text + '\n';
                }
              });
            } else if (step.type === 'model_output' && step.content) {
              step.content.forEach(block => {
                if (block.type === 'text' && block.text) {
                  fullText += block.text;
                }
              });
            }
          });
        }

        if (thoughtsText.trim()) {
          fullText = `<think>\n${thoughtsText.trim()}\n</think>\n\n` + fullText;
        }

        const openaiResponse = {
          id: `chatcmpl-${Date.now()}`,
          object: 'chat.completion',
          created: Math.floor(Date.now() / 1000),
          model: model,
          choices: [{
            index: 0,
            message: {
              role: 'assistant',
              content: fullText
            },
            finish_reason: 'stop'
          }],
          usage: {
            prompt_tokens: 0,
            completion_tokens: 0,
            total_tokens: 0
          }
        };
        res.json(openaiResponse);
      }
      return;
    }

    // --- NVIDIA NIM Pathway ---
    if (!activeApiKey || activeApiKey === 'dummy-key') {
      return res.status(401).json({
        error: { 
          message: 'API Key NVIDIA kosong. Silakan tempelkan API Key NVIDIA (nvapi-...) Anda di kolom API Key Janitor AI.', 
          type: 'invalid_request_error', 
          code: 401 
        }
      });
    }

    const config = getModelConfig(nimModel, ENABLE_THINKING_MODE);

    const nimRequest = {
      model: nimModel,
      messages: messages,
      temperature: temperature !== undefined ? temperature : 0.7,
      max_tokens: max_tokens ? Math.min(max_tokens, config.maxTokens) : config.maxTokens,
      chat_template_kwargs: config.chatTemplateKwargs,
      stream: stream || false
    };

    if (nimModel.toLowerCase().includes('mistral-medium-3.5')) {
      nimRequest.reasoning_effort = "high";
    }
    
    const response = await axios.post(`${NIM_API_BASE}/chat/completions`, nimRequest, {
      headers: {
        'Authorization': `Bearer ${activeApiKey}`,
        'Content-Type': 'application/json'
      },
      responseType: stream ? 'stream' : 'json'
    });
    
    if (stream) {
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');
      
      let buffer = '';
      let reasoningStarted = false;
      
      response.data.on('data', (chunk) => {
        buffer += chunk.toString();
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';
        
        lines.forEach(line => {
          if (line.startsWith('data: ')) {
            if (line.includes('[DONE]')) {
              res.write(line + '\n');
              return;
            }
            
            try {
              const data = JSON.parse(line.slice(6));
              if (data.choices?.[0]?.delta) {
                const reasoning = data.choices[0].delta.reasoning_content || data.choices[0].delta.reasoning;
                const content = data.choices[0].delta.content;
                
                if (SHOW_REASONING) {
                  let combinedContent = '';
                  
                  if (reasoning && !reasoningStarted) {
                    combinedContent = '<think>\n' + reasoning;
                    reasoningStarted = true;
                  } else if (reasoning) {
                    combinedContent = reasoning;
                  }
                  
                  if (content && reasoningStarted) {
                    combinedContent += '</think>\n\n' + content;
                    reasoningStarted = false;
                  } else if (content) {
                    combinedContent += content;
                  }
                  
                  if (combinedContent) {
                    data.choices[0].delta.content = combinedContent;
                    delete data.choices[0].delta.reasoning_content;
                    delete data.choices[0].delta.reasoning;
                  }
                } else {
                  if (content) {
                    data.choices[0].delta.content = content;
                  } else {
                    data.choices[0].delta.content = '';
                  }
                  delete data.choices[0].delta.reasoning_content;
                  delete data.choices[0].delta.reasoning;
                }
              }
              res.write(`data: ${JSON.stringify(data)}\n\n`);
            } catch (e) {
              res.write(line + '\n');
            }
          }
        });
      });
      
      response.data.on('end', () => res.end());
      response.data.on('error', (err) => {
        res.end();
      });
    } else {
      const openaiResponse = {
        id: `chatcmpl-${Date.now()}`,
        object: 'chat.completion',
        created: Math.floor(Date.now() / 1000),
        model: model,
        choices: response.data.choices.map(choice => {
          let fullContent = choice.message?.content || '';
          const reasoningText = choice.message?.reasoning_content || choice.message?.reasoning;
          
          if (SHOW_REASONING && reasoningText) {
            fullContent = '<think>\n' + reasoningText + '\n</think>\n\n' + fullContent;
          }
          
          return {
            index: choice.index,
            message: {
              role: choice.message.role,
              content: fullContent
            },
            finish_reason: choice.finish_reason
          };
        }),
        usage: response.data.usage || {
          prompt_tokens: 0,
          completion_tokens: 0,
          total_tokens: 0
        }
      };
      res.json(openaiResponse);
    }
  } catch (error) {
    res.status(error.response?.status || 500).json({
      error: {
        message: error.response?.data?.error?.message || error.response?.data?.message || error.message || 'Internal server error',
        type: 'invalid_request_error',
        code: error.response?.status || 500
      }
    });
  }
});

app.all('*', (req, res) => {
  res.status(404).json({
    error: { message: `Endpoint ${req.path} not found`, type: 'invalid_request_error', code: 404 }
  });
});

module.exports = app;
