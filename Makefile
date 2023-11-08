build_docker:
	docker compose build

run_docker: 
	docker compose run

exec_docker:
	docker compose exec -it hyperdiffusion bash