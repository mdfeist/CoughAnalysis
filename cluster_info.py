class ClusterInfo:
    def __init__(self, id):
        self._id = id
        self._dates = []

    def add_date(self, date):
        self._dates.append(date)

    def sort_dates(self):
        self._dates.sort(key=lambda x: x.date())

    def size(self):
        return len(self._dates)

    def __str__(self) -> str:
        return f'Cluster: {self._id} - Size: {len(self._dates)}'
