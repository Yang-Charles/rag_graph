const { createApp } = Vue;

createApp({
  data() {
    return { query: '', file: null, resp: null, chart: null, selected: ['fulltext','semantic','fused'], topk: 5, showCounts: true }
  },
  computed: {
    displayMethods() {
      // keep user order; default to selected
      return this.selected && this.selected.length ? this.selected : ['fulltext','semantic','image','kg','fused'];
    }
  },
  methods: {
    onFile(e) { this.file = e.target.files[0] },
    async onSearch() {
      const fd = new FormData();
      fd.append('query', this.query);
      if (this.file) fd.append('image', this.file);
      // include selected methods as hint (backend may ignore)
      fd.append('methods', this.selected.join(','));
      fd.append('topk', String(this.topk));
      const res = await fetch('/api/search', { method: 'POST', body: fd });
      this.resp = await res.json();
      this.drawChart();
    },
    formatItem(item) {
      if (!item) return '';
      if (Array.isArray(item)) return `${item[0]} (score: ${item[1]})`;
      if (typeof item === 'object') return JSON.stringify(item);
      return String(item);
    },
    drawChart() {
      if (!this.resp) return;
      const methods = this.displayMethods;
      const counts = {};
      methods.forEach(m => {
        const arr = this.resp[m] || [];
        arr.slice(0, this.topk).forEach(item => {
          const id = Array.isArray(item) ? item[0] : (item && item.kg_id ? item.kg_id : JSON.stringify(item));
          counts[id] = counts[id] ? counts[id] + 1 : 1;
        })
      })

      const labels = Object.keys(counts);
      const data = Object.values(counts);

      const ctx = document.getElementById('compareChart').getContext('2d');
      if (this.chart) this.chart.destroy();
      this.chart = new Chart(ctx, {
        type: 'bar',
        data: {
          labels: labels,
          datasets: [{ label: this.showCounts ? 'Presence count in top-k' : 'Presence', data: data, backgroundColor: 'rgba(54, 162, 235, 0.6)' }]
        },
        options: { responsive: true, maintainAspectRatio: false }
      });
    }
  }
}).mount('#app')
