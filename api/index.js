// index.js - OpenAI to NVIDIA NIM, Z.AI, OpenRouter, Mistral AI & DeepSeek Official Proxy
const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();

app.use(cors());
// Set incoming payload size limit to 4.5 MB to support up to 1M token contexts
app.use(express.json({ limit: '4.5mb' }));
app.use(express.urlencoded({ limit: '4.5mb', extended: true }));

const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY = process.env.NIM_API_KEY;
const OPENROUTER_API_KEY = process.env.OPENROUTER_API_KEY;
const MISTRAL_API_KEY = process.env.MISTRAL_API_KEY;
const DEEPSEEK_API_KEY = process.env.DEEPSEEK_API_KEY;
const ZAI_API_KEY = process.env.ZAI_API_KEY;

const MODEL_MAPPING = {
  // 1. NVIDIA NIM Models (User Custom Order)
  'claude-3-opus': 'z-ai/glm-5.2',                            // 1. GLM 5.2 (Default)
  'minimax-m3': 'minimaxai/minimax-m3',                        // 2. MiniMax M3
  'gpt-4-turbo': 'google/gemma-4-31b-it',                      // 3. Gemma 31B
  'nemotron-3-ultra': 'nvidia/nemotron-3-ultra-550b-a55b',     // 4. Nemotron Ultra
  'mistral-medium-3.5': 'mistralai/mistral-medium-3.5-128b',   // 5. Mistral 3.5 NIM
  'step-3.7-flash': 'stepfun-ai/step-3.7-flash',               // 6. Step Flash
  'nemotron-3-super': 'nvidia/nemotron-3-super-120b-a12b',     // 7. Nemotron Super
  'muse-glimmer': 'meta/muse-glimmer-30b',                    // 8. Muse Glimmer 30B
  'gpt-3.5-turbo': 'thinkingmachines/inkling',                 // 9. Inkling
  'laguna-xs-2.1': 'poolside/laguna-xs-2.1',                   // 10. Laguna
  'gemini-pro': 'google/diffusiongemma-26b-a4b-it',             // 11. DiffusionGemma 26B

  // 2. Z.ai Official API Direct Models (Free GLM)
  'glm-4.7-flash': 'glm-4.7-flash',
  'glm-4.5-flash': 'glm-4.5-flash',

  // 3. Official DeepSeek API Direct Models
  'deepseek-v4-flash': 'deepseek-v4-flash',
  'deepseek-v4-pro': 'deepseek-v4-pro',
  'deepseek-chat': 'deepseek-chat',

  // 4. OpenRouter Free Gemma Models
  'gemma-4-31b': 'google/gemma-4-31b-it:free',
  'gemma-4-26b': 'google/gemma-4-26b-a4b-it:free',

  // 5. Mistral AI Official API Models
  'mistral-large-2512': 'mistral-large-latest',
  'mistral-medium-2508': 'mistral-medium-3-5'
};

function getModelConfig(nimModel, enableThinking) {
  let maxTokens = 16384; 
  let chatTemplateKwargs = undefined;
  let reasoningBudget = undefined;
  let tempOverride = undefined;
  let topPOverride = undefined;
  let seedOverride = undefined;

  const modelLower = nimModel.toLowerCase();

  if (modelLower.includes('nemotron-3-ultra')) {
    maxTokens = 16384;
    tempOverride = 1.0;
    topPOverride = 0.95;
    if (enableThinking) {
      chatTemplateKwargs = { enable_thinking: true };
      reasoningBudget = 16384;
    } else {
      chatTemplateKwargs = { enable_thinking: false };
    }
  }
  else if (modelLower.includes('nemotron-3-super')) {
    maxTokens = 16384;
    tempOverride = 1.0;
    topPOverride = 0.95;
    if (enableThinking) {
      chatTemplateKwargs = { enable_thinking: true };
      reasoningBudget = 16384;
    } else {
      chatTemplateKwargs = { enable_thinking: false };
    }
  }
  else if (modelLower.includes('step-3.7-flash')) {
    maxTokens = 16384;
    tempOverride = 1.0;
    topPOverride = 0.95;
    seedOverride = 42;
    // StepFun streams reasoning_content natively in delta.content
  }
  else if (modelLower.includes('muse-glimmer')) {
    maxTokens = 8192;
    tempOverride = 1.0;
    topPOverride = 0.95;
    // Reasoning strength is injected dynamically into system prompt for Muse Glimmer
  }
  else if (modelLower.includes('laguna-xs')) {
    maxTokens = 8192;
    tempOverride = 1.0;
    topPOverride = 0.95;
  }
  else if (modelLower.includes('minimax-m3') || modelLower.includes('minimax-3')) {
    maxTokens = 8192;
    chatTemplateKwargs = enableThinking ? { thinking_mode: "enabled" } : { thinking_mode: "disabled" };
  }
  else if (modelLower.includes('gemma-4')) {
    maxTokens = 16384;
    chatTemplateKwargs = enableThinking ? { enable_thinking: true } : { enable_thinking: false };
  } 
  else if (modelLower.includes('glm-5.2')) {
    maxTokens = 16384;
    chatTemplateKwargs = enableThinking ? { thinking: { type: "enabled" }, reasoning_effort: "high" } : { thinking: { type: "disabled" } };
  }
  else if (modelLower.includes('diffusiongemma') || modelLower.includes('gemma-26b')) {
    maxTokens = 4096;
    tempOverride = 1.0;
    topPOverride = 0.95;
    if (enableThinking) {
      chatTemplateKwargs = { enable_thinking: true };
    }
  }
  else if (modelLower.includes('inkling')) {
    maxTokens = 8192;
    chatTemplateKwargs = undefined; 
  }

  return { maxTokens, chatTemplateKwargs, reasoningBudget, tempOverride, topPOverride, seedOverride };
}

app.get('/health', (req, res) => {
  res.json({ 
    status: 'ok', 
    service: 'OpenAI to NVIDIA NIM, Z.AI, OpenRouter, Mistral AI & DeepSeek Official Proxy',
    dynamic_thinking: true
  });
});

app.get('/v1/models', (req, res) => {
  const models = Object.keys(MODEL_MAPPING).map(model => ({
    id: model,
    object: 'model',
    created: Date.now(),
    owned_by: 'multi-provider-proxy'
  }));
  res.json({ object: 'list', data: models });
});

app.post('/v1/chat/completions', async (req, res) => {
  try {
    const { model, messages, temperature, max_tokens, stream, forbiddenWords, frequency_penalty, enableThinking } = req.body;
    
    // Read enableThinking parameter from Client UI (Default: true)
    const isThinkingEnabled = enableThinking !== undefined ? Boolean(enableThinking) : true;

    const authHeader = req.headers.authorization;
    const clientApiKey = authHeader ? authHeader.replace('Bearer ', '').trim() : '';

    let targetModel = MODEL_MAPPING[model] || model;

    // 1. Raw Messages
    let rawMessages = JSON.parse(JSON.stringify(messages));

    // 2. CONTEXT OPTIMIZATION: Clean <think>...</think> tags from past assistant history
    // FIX FOR HTTP 400: Never allow assistant content to be an empty string ("")
    let processedMessages = rawMessages.map(m => {
      if (m.role === 'assistant' && typeof m.content === 'string') {
        let cleanContent = m.content.replace(/<think>([\s\S]*?)(?:<\/think>|$)/gi, '').trim();
        if (!cleanContent) {
          cleanContent = "(thought process completed)";
        }
        return { role: m.role, content: cleanContent };
      }
      return m;
    });

    // 3. Construct dynamic Negative Constraint Directive if forbiddenWords are set
    let negativeConstraint = "";
    if (forbiddenWords && typeof forbiddenWords === 'string' && forbiddenWords.trim().length > 0) {
      negativeConstraint = `\n\n[CRITICAL NEGATIVE CONSTRAINTS: You are strictly forbidden from outputting or using any of the following words, phrases, AI clichés, or behaviors: ${forbiddenWords.trim()}. Choose natural, creative alternatives and adhere strictly to these exclusions.]`;
    }

    // 4. Prompt-Forced Thinking for Mistral Large
    let forcedThinkingDirective = "";
    if (targetModel === 'mistral-large-latest' && isThinkingEnabled) {
      forcedThinkingDirective = `\n\n[REASONING DIRECTIVE: Before providing your final answer, write out your detailed step-by-step thinking process inside <think>...</think> tags.]`;
    }

    // 5. Reasoning Strength Control for Muse Glimmer 30B
    let museGlimmerDirective = "";
    if (targetModel.toLowerCase().includes('muse-glimmer')) {
      museGlimmerDirective = isThinkingEnabled ? "\n\n[Reasoning strength: high]" : "\n\n[Reasoning strength: low]";
    }

    // Append extra system instructions
    const extraSystemInstructions = negativeConstraint + forcedThinkingDirective + museGlimmerDirective;
    if (extraSystemInstructions.trim()) {
      const sysMsgIndex = processedMessages.findIndex(m => m.role === 'system');
      if (sysMsgIndex !== -1) {
        processedMessages[sysMsgIndex].content += extraSystemInstructions;
      } else {
        processedMessages.unshift({ role: 'system', content: extraSystemInstructions.trim() });
      }
    }

    // Determine Provider Routing
    const isZaiAPI = targetModel === 'glm-4.7-flash' || targetModel === 'glm-4.5-flash';
    const isOfficialDeepSeek = targetModel === 'deepseek-v4-flash' || targetModel === 'deepseek-v4-pro' || targetModel === 'deepseek-chat';
    const isMistralAPI = targetModel.startsWith('mistral-large') || targetModel.startsWith('mistral-medium-3-5');
    const isOpenRouterAPI = targetModel.includes(':free') || targetModel.startsWith('google/');

    let requestUrl = `${NIM_API_BASE}/chat/completions`;
    let requestHeaders = {
      'Authorization': `Bearer ${NIM_API_KEY || clientApiKey}`,
      'Content-Type': 'application/json'
    };
    let requestPayload = {};

    // --- 1. Z.AI Official API Route (Free GLM Flash) ---
    if (isZaiAPI) {
      const activeZaiKey = ZAI_API_KEY || clientApiKey;
      if (!activeZaiKey) {
        return res.status(401).json({
          error: { message: 'ZAI_API_KEY is missing in Vercel environment variables.', type: 'invalid_request_error', code: 401 }
        });
      }
      requestUrl = 'https://api.z.ai/api/paas/v4/chat/completions';
      requestHeaders = {
        'Authorization': `Bearer ${activeZaiKey}`,
        'Content-Type': 'application/json'
      };

      requestPayload = {
        model: targetModel,
        messages: processedMessages,
        temperature: temperature !== undefined ? temperature : 0.7,
        max_tokens: max_tokens || 8192,
        stream: stream || false
      };

      if (isThinkingEnabled) {
        requestPayload.thinking = {
          type: "enabled",
          clear_thinking: false
        };
      } else {
        requestPayload.thinking = {
          type: "disabled"
        };
      }
    }
    // --- 2. Official DeepSeek API Route ---
    else if (isOfficialDeepSeek) {
      const activeDeepSeekKey = DEEPSEEK_API_KEY || clientApiKey;
      if (!activeDeepSeekKey) {
        return res.status(401).json({
          error: { message: 'DEEPSEEK_API_KEY is missing in Vercel environment variables.', type: 'invalid_request_error', code: 401 }
        });
      }
      requestUrl = 'https://api.deepseek.com/chat/completions';
      requestHeaders = {
        'Authorization': `Bearer ${activeDeepSeekKey}`,
        'Content-Type': 'application/json'
      };

      requestPayload = {
        model: targetModel,
        messages: processedMessages,
        temperature: temperature !== undefined ? temperature : 0.7,
        max_tokens: max_tokens || 8192,
        stream: stream || false
      };
    }
    // --- 3. Mistral AI Official API Route ---
    else if (isMistralAPI) {
      if (!MISTRAL_API_KEY) {
        return res.status(401).json({
          error: { message: 'MISTRAL_API_KEY is missing in Vercel environment variables.', type: 'invalid_request_error', code: 401 }
        });
      }
      requestUrl = 'https://api.mistral.ai/v1/chat/completions';
      requestHeaders['Authorization'] = `Bearer ${MISTRAL_API_KEY}`;

      requestPayload = {
        model: targetModel,
        messages: processedMessages,
        temperature: temperature !== undefined ? temperature : 0.7,
        max_tokens: max_tokens || 16384,
        stream: stream || false
      };

      if (targetModel.includes('medium') && isThinkingEnabled) {
        requestPayload.reasoning_effort = "high";
      } else {
        if (frequency_penalty !== undefined) {
          requestPayload.frequency_penalty = frequency_penalty;
        }
      }
    }
    // --- 4. OpenRouter API Route ---
    else if (isOpenRouterAPI) {
      if (!OPENROUTER_API_KEY) {
        return res.status(401).json({
          error: { message: 'OPENROUTER_API_KEY is missing in Vercel environment variables.', type: 'invalid_request_error', code: 401 }
        });
      }
      requestUrl = 'https://openrouter.ai/api/v1/chat/completions';
      requestHeaders = {
        'Authorization': `Bearer ${OPENROUTER_API_KEY}`,
        'HTTP-Referer': 'https://tes-tis-tes.vercel.app',
        'X-Title': 'Private Chatbot Client',
        'Content-Type': 'application/json'
      };

      requestPayload = {
        model: targetModel,
        messages: processedMessages,
        temperature: temperature !== undefined ? temperature : 0.7,
        max_tokens: max_tokens || 16384,
        stream: stream || false
      };

      if (isThinkingEnabled) {
        requestPayload.reasoning = { effort: "high" };
        requestPayload.include_reasoning = true;
      }
    }
    // --- 5. NVIDIA NIM API Route ---
    else {
      const activeApiKey = NIM_API_KEY || clientApiKey;
      if (!activeApiKey || activeApiKey === 'dummy-key') {
        return res.status(401).json({
          error: { message: 'NVIDIA API Key is missing.', type: 'invalid_request_error', code: 401 }
        });
      }

      const config = getModelConfig(targetModel, isThinkingEnabled);
      requestPayload = {
        model: targetModel,
        messages: processedMessages,
        temperature: config.tempOverride !== undefined ? config.tempOverride : (temperature !== undefined ? temperature : 0.7),
        max_tokens: max_tokens ? Math.min(max_tokens, config.maxTokens) : config.maxTokens,
        stream: stream || false
      };

      if (config.topPOverride !== undefined) {
        requestPayload.top_p = config.topPOverride;
      }

      if (config.seedOverride !== undefined) {
        requestPayload.seed = config.seedOverride;
      }

      if (config.chatTemplateKwargs) {
        requestPayload.chat_template_kwargs = config.chatTemplateKwargs;
      }

      if (config.reasoningBudget) {
        requestPayload.reasoning_budget = config.reasoningBudget;
      }

      if (targetModel.toLowerCase().includes('mistral-medium-3.5') && isThinkingEnabled) {
        requestPayload.reasoning_effort = "high";
      }
    }

    // Execute API Request
    const response = await axios.post(requestUrl, requestPayload, {
      headers: requestHeaders,
      responseType: stream ? 'stream' : 'json'
    });
    
    // --- Unified Multi-Provider Streaming Handler ---
    if (stream) {
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');
      
      let buffer = '';
      let reasoningStarted = false;
      let stepfunThoughtStarted = false;
      const isStepFun = targetModel.toLowerCase().includes('step-3.7-flash');
      
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
              const deltaChoice = data.choices?.[0]?.delta;

              if (deltaChoice) {
                let reasoningText = "";
                let contentText = "";

                if (deltaChoice.reasoning_content) reasoningText += deltaChoice.reasoning_content;
                if (deltaChoice.reasoning) reasoningText += deltaChoice.reasoning;

                if (Array.isArray(deltaChoice.content)) {
                  deltaChoice.content.forEach(item => {
                    if (item.type === 'thinking' && item.thinking) {
                      item.thinking.forEach(t => { if (t.text) reasoningText += t.text; });
                    } else if (item.type === 'text' && item.text) {
                      contentText += item.text;
                    }
                  });
                } else if (typeof deltaChoice.content === 'string') {
                  contentText += deltaChoice.content;
                }

                // STEPFUN STREAM INTERCEPTOR
                if (isThinkingEnabled && isStepFun && !stepfunThoughtStarted && contentText.trim().length > 0) {
                  if (!contentText.startsWith('<think>')) {
                    contentText = '<think>\n' + contentText;
                  }
                  stepfunThoughtStarted = true;
                }

                // Format & wrap into <think> ... </think> tags IF THINKING MODE IS ACTIVE
                if (isThinkingEnabled) {
                  let combined = '';
                  
                  if (reasoningText && !reasoningStarted) {
                    combined = '<think>\n' + reasoningText;
                    reasoningStarted = true;
                  } else if (reasoningText) {
                    combined = reasoningText;
                  }
                  
                  if (contentText && reasoningStarted) {
                    combined += '\n</think>\n\n' + contentText;
                    reasoningStarted = false;
                  } else if (contentText) {
                    combined += contentText;
                  }
                  
                  if (combined) {
                    data.choices[0].delta.content = combined;
                    delete data.choices[0].delta.reasoning_content;
                    delete data.choices[0].delta.reasoning;
                  }
                } else {
                  // If thinking is off, strip reasoning text and forward content only
                  if (isStepFun && contentText.includes('</think>')) {
                    const parts = contentText.split('</think>');
                    contentText = parts[parts.length - 1].trim();
                  }
                  data.choices[0].delta.content = contentText;
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
      response.data.on('error', (err) => res.end());
    } else {
      // Non-streaming response parsing
      let fullContent = '';
      let reasoningText = '';

      const choice = response.data.choices[0];
      if (choice) {
        if (Array.isArray(choice.message?.content)) {
          choice.message.content.forEach(item => {
            if (item.type === 'thinking' && item.thinking) {
              item.thinking.forEach(t => { if (t.text) reasoningText += t.text; });
            } else if (item.type === 'text' && item.text) {
              fullContent += item.text;
            }
          });
        } else if (typeof choice.message?.content === 'string') {
          fullContent = choice.message.content;
        }

        if (choice.message?.reasoning_content) reasoningText += choice.message.reasoning_content;
        if (choice.message?.reasoning) reasoningText += choice.message.reasoning;
      }

      if (isThinkingEnabled && reasoningText.trim()) {
        fullContent = `<think>\n${reasoningText.trim()}\n</think>\n\n` + fullContent;
      }

      const openaiResponse = {
        id: `chatcmpl-${Date.now()}`,
        object: 'chat.completion',
        created: Math.floor(Date.now() / 1000),
        model: model,
        choices: [{
          index: 0,
          message: { role: 'assistant', content: fullContent },
          finish_reason: 'stop'
        }],
        usage: response.data.usage || { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 }
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
