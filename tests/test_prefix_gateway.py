from app.load_balancer import LoadBalancer, ServerInfo


def test_prefix_hash_is_stable():
    prompt = "shared prompt prefix" * 100

    assert LoadBalancer.compute_prefix_hash(prompt) == LoadBalancer.compute_prefix_hash(prompt)
    assert LoadBalancer.compute_prefix_hash(prompt) != LoadBalancer.compute_prefix_hash("different" + prompt)


def test_cache_aware_routing_prefers_healthy_prefix_owner():
    balancer = LoadBalancer()
    server_a = ServerInfo("node-a", 8000, "now")
    server_b = ServerInfo("node-b", 8001, "now")
    balancer.servers = {
        "node-a:8000": server_a,
        "node-b:8001": server_b,
    }

    prefix_hash = LoadBalancer.compute_prefix_hash("same prefix")
    first_server, first_hit, _ = balancer.get_cache_aware_server(prefix_hash)
    balancer.record_prefix_owner(prefix_hash, first_server)

    second_server, second_hit, owner_key = balancer.get_cache_aware_server(prefix_hash)

    assert first_hit is False
    assert second_hit is True
    assert second_server is first_server
    assert owner_key == f"{first_server.hostname}:{first_server.port}"


def test_cache_aware_routing_falls_back_when_owner_unhealthy():
    balancer = LoadBalancer()
    server_a = ServerInfo("node-a", 8000, "now")
    server_b = ServerInfo("node-b", 8001, "now")
    balancer.servers = {
        "node-a:8000": server_a,
        "node-b:8001": server_b,
    }
    prefix_hash = LoadBalancer.compute_prefix_hash("same prefix")
    balancer.record_prefix_owner(prefix_hash, server_a)
    server_a.is_healthy = False

    selected, cache_hit, owner_key = balancer.get_cache_aware_server(prefix_hash)

    assert selected is server_b
    assert cache_hit is False
    assert owner_key == "node-a:8000"
