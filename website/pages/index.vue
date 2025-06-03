<template>
  <div class="min-h-screen bg-gray-100 p-6">
    <div class="max-w-6xl mx-auto">
      <h1 class="text-3xl font-bold text-gray-800 mb-6">Real-time Car Predictions</h1>

      <!-- Connection Status -->
      <div class="bg-white rounded-lg shadow-md p-4 mb-6">
        <div class="flex items-center gap-4">
          <div class="flex items-center gap-2">
            <div
                :class="[
                'w-3 h-3 rounded-full',
                connectionStatus === 'connected' ? 'bg-green-500' :
                connectionStatus === 'connecting' ? 'bg-yellow-500' : 'bg-red-500'
              ]"
            ></div>
            <span class="font-medium">
              {{ connectionStatus === 'connected' ? 'Connected' :
                connectionStatus === 'connecting' ? 'Connecting...' : 'Disconnected' }}
            </span>
          </div>

          <button
              @click="toggleConnection"
              :disabled="connectionStatus === 'connecting'"
              class="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 disabled:opacity-50"
          >
            {{ connectionStatus === 'connected' ? 'Disconnect' : 'Connect' }}
          </button>

          <div class="text-sm text-gray-600">
            Predictions received: {{ predictionCount }}
          </div>
        </div>
      </div>

      <div class="grid lg:grid-cols-2 gap-6">
        <!-- Car Status Visualization -->
        <div class="lg:col-span-1">
          <CarStatus
              :predictions="parsedPredictions"
              :lastUpdated="latestPrediction?.timestamp"
          />
        </div>

        <!-- Raw Prediction Data -->
        <div class="lg:col-span-1">
          <div class="bg-white rounded-lg shadow-md p-6">
            <h2 class="text-xl font-semibold mb-4">Raw Prediction Data</h2>

            <div v-if="latestPrediction" class="space-y-4">
              <div class="grid grid-cols-2 gap-4 text-sm">
                <div>
                  <span class="font-medium">Timestamp:</span>
                  {{ formatTimestamp(latestPrediction.timestamp) }}
                </div>
                <div>
                  <span class="font-medium">Frame Shape:</span>
                  {{ latestPrediction.frame_shape?.join(' × ') || 'N/A' }}
                </div>
              </div>

              <div>
                <h3 class="font-medium mb-2">Raw Predictions:</h3>
                <div class="bg-gray-50 p-4 rounded-lg">
                  <pre class="text-sm overflow-auto">{{ JSON.stringify(latestPrediction.predictions, null, 2) }}</pre>
                </div>
              </div>
            </div>

            <div v-else class="text-gray-500 text-center py-8">
              No predictions received yet
            </div>
          </div>
        </div>
      </div>


    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted } from 'vue'

// Page metadata
useHead({
  title: 'Car Door Predictions',
  meta: [
    { name: 'description', content: 'Real-time car door status predictions' }
  ]
})

// Reactive state
const connectionStatus = ref('disconnected')
const predictionCount = ref(0)
const latestPrediction = ref(null)

const websocket = ref(null)

// WebSocket URL
const wsUrl = 'ws://localhost:8000/ws/predictions'

// Computed property to parse predictions into the expected format
const parsedPredictions = computed(() => {
  if (!latestPrediction.value?.predictions) return {}

  // Handle different prediction formats
  const predictions = latestPrediction.value.predictions

  // If predictions is already in the right format
  if (typeof predictions === 'object' && predictions.front_left_door !== undefined) {
    return predictions
  }

  // If predictions is an array or tensor, you might need to process it
  // This is a placeholder - adjust based on your actual model output
  if (Array.isArray(predictions)) {
    // Example processing - replace with your actual logic
    return {
      front_left_door: predictions[0] > 0.5 ? 'open' : 'closed',
      front_right_door: predictions[1] > 0.5 ? 'open' : 'closed',
      rear_left_door: predictions[2] > 0.5 ? 'open' : 'closed',
      rear_right_door: predictions[3] > 0.5 ? 'open' : 'closed',
      hood: predictions[4] > 0.5 ? 'open' : 'closed'
    }
  }

  return {}
})

// WebSocket connection management
const connectWebSocket = () => {
  if (websocket.value) {
    websocket.value.close()
  }

  connectionStatus.value = 'connecting'

  try {
    websocket.value = new WebSocket(wsUrl)

    websocket.value.onopen = () => {
      connectionStatus.value = 'connected'
      console.log('Connected to prediction WebSocket')

      // Send ping to keep connection alive
      setInterval(() => {
        if (websocket.value?.readyState === WebSocket.OPEN) {
          websocket.value.send(JSON.stringify({ type: 'ping' }))
        }
      }, 30000)
    }

    websocket.value.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data)

        if (data.type === 'pong') {
          return // Ignore pong responses
        }

        // Handle prediction data
        if (data.timestamp && data.predictions) {
          latestPrediction.value = data
          predictionCount.value++
        }
      } catch (error) {
        console.error('Error parsing WebSocket message:', error)
      }
    }

    websocket.value.onclose = () => {
      connectionStatus.value = 'disconnected'
      console.log('Disconnected from prediction WebSocket')
    }

    websocket.value.onerror = (error) => {
      connectionStatus.value = 'disconnected'
      console.error('WebSocket error:', error)
    }
  } catch (error) {
    connectionStatus.value = 'disconnected'
    console.error('Failed to create WebSocket connection:', error)
  }
}

const disconnectWebSocket = () => {
  if (websocket.value) {
    websocket.value.close()
    websocket.value = null
  }
  connectionStatus.value = 'disconnected'
}

const toggleConnection = () => {
  if (connectionStatus.value === 'connected') {
    disconnectWebSocket()
  } else {
    connectWebSocket()
  }
}

const clearHistory = () => {
  predictionCount.value = 0
  latestPrediction.value = null
}

const formatTimestamp = (timestamp) => {
  if (!timestamp) return 'N/A'
  return new Date(timestamp * 1000).toLocaleTimeString()
}

// Lifecycle hooks
onMounted(() => {
  connectWebSocket()
})

onUnmounted(() => {
  disconnectWebSocket()
})
</script>

<style scoped>
/* Custom scrollbar styles removed as no longer needed */
</style>