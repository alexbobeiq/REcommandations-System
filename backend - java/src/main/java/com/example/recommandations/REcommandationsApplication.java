package com.example.recommandations;

import org.springframework.boot.CommandLineRunner;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.context.annotation.Bean;
import org.springframework.core.ParameterizedTypeReference;
import org.springframework.web.client.RestClient;

import java.util.List;

@SpringBootApplication
public class REcommandationsApplication {

    public static void main(String[] args) {
        SpringApplication.run(REcommandationsApplication.class, args);
    }

    @Bean
    CommandLineRunner initDatabase(ProductRepository repository, RestClient.Builder restClientBuilder) {
        return args -> {
            repository.deleteAll();

            RestClient pythonClient = restClientBuilder.baseUrl("http://127.0.0.1:8000").build();

            try {
                List<Product> catalogComplet = pythonClient.get()
                        .uri("/catalog")
                        .retrieve()
                        .body(new ParameterizedTypeReference<List<Product>>() {});

                if (catalogComplet != null) {
                    repository.saveAll(catalogComplet);
                    System.out.println("Succes " + catalogComplet.size() + " products imported");
                }
            } catch (Exception e) {
                System.out.println("error: " + e.getMessage());
                repository.save(new Product("85123A", "Fallback Product", 10.0));
            }
        };
    }
}

