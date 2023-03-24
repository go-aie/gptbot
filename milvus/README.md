# Milvus

Using [Milvus][1] as the vector store.


## Installation

For how to install Milvus in different scenarios, see [Official Documents][2].

For testing purpose, here we choose to [Install Milvus Standalone with Docker Compose][3].

```bash
$ wget https://github.com/milvus-io/milvus/releases/download/v2.2.4/milvus-standalone-docker-compose.yml -O docker-compose.yml
$ sudo docker compose up -d
```

## Testing

```bash
$ go test -v -race
=== RUN   TestMilvus_Query
--- PASS: TestMilvus_Query (7.34s)
PASS
ok  	github.com/go-aie/gptbot/milvus	7.866s
```


[1]: https://milvus.io/
[2]: https://milvus.io/docs/install_standalone-operator.md
[3]: https://milvus.io/docs/install_standalone-docker.md