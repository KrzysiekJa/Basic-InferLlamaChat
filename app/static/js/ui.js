// UI scripts moved from template
// Depends on Showdown being loaded before this script (include with defer)

const converter = new showdown.Converter({
    tables: true,
    ghCodeBlocks: true,
    simplifiedAutoLink: true,
    strikethrough: true,
    tasklists: true,
})

function renderMarkdown(raw) {
    let text = raw ?? ''

    try {
        text = JSON.parse(text)
    } catch (e) {
        // not JSON
    }

    text = text.replace(/\\n/g, "\n")
    text = text.replace(/\\+/g, "")

    // Normalize asterisk lists to dash lists and ensure blank line
    text = text.replace(/(^|\n)[ \t]*\*[ \t]+/g, "$1- ")
    text = text.replace(/([^\n])\n- /g, "$1\n\n- ")

    return converter.makeHtml(text)
}

async function handleSubmit(event, formType = "batch") {
    event.preventDefault()
    const form = event.target
    const formData = new FormData(form)
    const userPrompt = formData.get('user-prompt')
    const maxTokens = parseInt(formData.get('max-tokens'), 10)
    const endpoint = '/api/v1/predict/' + formType
    const responseContainerId = formType + '-response'

    if (formType === "stream") {
        document.getElementById(responseContainerId).innerHTML = ''
    }

    try {
        const response = await fetch(endpoint, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                user_prompt: userPrompt,
                max_tokens: maxTokens,
            }),
        })

        if (!response.ok) {
            const errorText = await response.text()
            console.error(`HTTP Error: ${response.status} - ${errorText}`)
            document.getElementById(responseContainerId).innerText = `Error: ${response.status} - ${errorText}`
            return
        }

        if (formType === "stream") {
            const reader = response.body.getReader()
            await readStream(reader, responseContainerId)
        } else {
            const text = await response.text()
            setBatchText(text, responseContainerId)
        }
    } catch (err) {
        console.error("Request error:", err)
        document.getElementById(responseContainerId).innerText = `Error: ${err.message}`
    }
}

function setBatchText(text, containerId) {
    const html = renderMarkdown(text)
    document.getElementById(containerId).innerHTML = html
}

async function readStream(reader, containerId) {
    const decoder = new TextDecoder()
    let buffer = ''

    while (true) {
        const { done, value } = await reader.read()

        if (done) break

        buffer += decoder.decode(value, { stream: true })

        // Render accumulated markdown so far
        document.getElementById(containerId).innerHTML = renderMarkdown(buffer)
    }

    document.getElementById(containerId).innerHTML = renderMarkdown(buffer)
}

document.addEventListener('DOMContentLoaded', () => {
    const batchForm = document.getElementById('batch-chat-form')
    const streamForm = document.getElementById('stream-chat-form')
    const weatherForm = document.getElementById('weather-chat-form')

    if (batchForm) batchForm.addEventListener('submit', (e) => { handleSubmit(e) })
    if (streamForm) streamForm.addEventListener('submit', (e) => { handleSubmit(e, 'stream') })
    if (weatherForm) weatherForm.addEventListener('submit', (e) => { handleSubmit(e, 'weather') })
})
