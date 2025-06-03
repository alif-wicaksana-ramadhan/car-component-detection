<template>
  <div class="min-h-screen bg-gray-200 flex items-center justify-center p-6">
    <div class="w-full max-w-md flex flex-col">
      <!-- Blue Button Header -->
      <div class="flex mb-6">
        <button
            @click="generateCaption"
            :disabled="!selectedImage || isLoading"
            class="flex-1 flex items-center justify-center cursor-pointer px-6 py-3 bg-blue-500 text-white text-lg font-medium rounded-lg hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors shadow-md"
        >
          <span v-if="isLoading" class="flex items-center">
            <svg class="animate-spin -ml-1 mr-2 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
              <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
              <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
            </svg>
            Processing...
          </span>
          <span v-else>Capture and Describe</span>
        </button>
      </div>

      <!-- White Card Container -->
      <div class="bg-white rounded-xl shadow-lg p-6 flex flex-col">
        <!-- Image Section -->
        <div class="flex flex-col mb-6">
          <!-- File Input -->
          <input
              ref="fileInput"
              type="file"
              accept="image/*"
              @change="handleFileSelect"
              class="hidden"
              :disabled="isLoading"
          />

          <!-- Image Display/Upload Area -->
          <div
              @click="triggerFileInput"
              @dragover.prevent
              @drop.prevent="handleFileDrop"
              :class="[
              'flex-1 rounded-lg overflow-hidden cursor-pointer transition-all hover:bg-gray-50',
              !selectedImage ? 'border-2 border-dashed border-gray-300 hover:border-blue-400 p-2 flex items-center justify-center' : 'flex'
            ]"
              @dragenter="isDragging = true"
              @dragleave="isDragging = false"
          >
            <!-- Image Preview -->
            <div v-if="selectedImage" class="relative flex-1">
              <img
                  :src="selectedImage"
                  alt="Selected image"
                  class="w-full h-64 object-cover rounded-lg"
              />
              <button
                  @click.stop="clearImage"
                  class="absolute top-2 right-2 flex items-center justify-center bg-red-500 text-white rounded-full w-6 h-6 hover:bg-red-600 text-sm"
              >
                ×
              </button>
            </div>

            <!-- Upload Placeholder -->
            <div v-else class="flex flex-col items-center justify-center text-gray-500 py-2 hover:text-gray-700 transition-colors">
              <p class="text-sm">📷 Click to upload image</p>
            </div>
          </div>
        </div>

        <!-- Description Text -->
        <div class="flex justify-center">
          <div v-if="caption" class="text-gray-600 text-lg leading-relaxed text-center">
            {{ caption }}
          </div>

          <div v-else-if="error" class="text-red-500 text-lg text-center">
            {{ error }}
          </div>

          <div v-else class="text-gray-400 text-lg text-center">
            Upload an image and click "Capture and Describe" to generate description
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'

// Page metadata
useHead({
  title: 'Capture and Describe',
  meta: [
    { name: 'description', content: 'AI-powered image captioning service' }
  ]
})

// Reactive state
const fileInput = ref(null)
const selectedImage = ref(null)
const selectedFile = ref(null)
const caption = ref('')
const isLoading = ref(false)
const error = ref('')
const isDragging = ref(false)

// File handling methods
const triggerFileInput = () => {
  fileInput.value?.click()
}

const handleFileSelect = (event) => {
  const file = event.target.files[0]
  if (file) {
    processFile(file)
  }
}

const handleFileDrop = (event) => {
  isDragging.value = false
  const file = event.dataTransfer.files[0]
  if (file && file.type.startsWith('image/')) {
    processFile(file)
  }
}

const processFile = (file) => {
  // Validate file size (10MB limit)
  if (file.size > 10 * 1024 * 1024) {
    error.value = 'File size must be less than 10MB'
    return
  }

  selectedFile.value = file

  // Create preview URL
  const reader = new FileReader()
  reader.onload = (e) => {
    selectedImage.value = e.target.result
    caption.value = ''
    error.value = ''
  }
  reader.readAsDataURL(file)
}

const clearImage = () => {
  selectedImage.value = null
  selectedFile.value = null
  caption.value = ''
  error.value = ''
  if (fileInput.value) {
    fileInput.value.value = ''
  }
}

const generateCaption = async () => {
  if (!selectedFile.value) {
    error.value = 'Please select an image first'
    return
  }

  isLoading.value = true
  error.value = ''
  caption.value = ''

  try {
    const formData = new FormData()
    formData.append('file', selectedFile.value)

    const response = await fetch('http://localhost:8000/api/caption-image', {
      method: 'POST',
      body: formData
    })

    if (!response.ok) {
      const errorData = await response.json()
      throw new Error(errorData.detail || 'Failed to generate caption')
    }

    const result = await response.json()
    caption.value = result.caption

  } catch (err) {
    error.value = err.message || 'An error occurred while generating the caption'
    console.error('Caption generation error:', err)
  } finally {
    isLoading.value = false
  }
}
</script>