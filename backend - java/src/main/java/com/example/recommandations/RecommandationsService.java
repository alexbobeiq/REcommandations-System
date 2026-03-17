package com.example.recommandations;

import org.springframework.stereotype.Service;
import org.springframework.web.client.HttpClientErrorException;
import org.springframework.web.client.RestClient;


import java.util.List;

@Service
public class RecommandationsService {
    private final RestClient restClient;
    private final ProductRepository productRepository;

    public RecommandationsService(RestClient.Builder restClientBuilder, ProductRepository productRepository) {
        this.restClient = restClientBuilder
                .baseUrl("http://127.0.0.1:8000")
                .build();
        this.productRepository = productRepository;
    }

    public List<String> getRecommandations(Integer customerID) {
        try {
            RecomandationsResponse response = restClient.get()
                    .uri("/recommandations/{customerID}", customerID)
                    .retrieve()
                    .body(RecomandationsResponse.class);
            if (response != null && response.recommandations() != null) {
                return response.recommandations();
            }
            return  List.of();
        } catch (HttpClientErrorException e) {
            System.out.println("Client nou detectat (" + customerID + "). AI-ul nu are istoric.");
            return List.of("BESTSELLER_CODE_1", "BESTSELLER_CODE_2", "BESTSELLER_CODE_3");
        } catch (Exception e) {
            System.out.println(e.getMessage());
            return List.of();
        }

    }

    public List<Product> getRecommandationsProducts(Integer customerID) {
        List<String> codes = getRecommandations(customerID);

        if (codes.isEmpty()) {
            return List.of();
        }

        return productRepository.findAllByCodeIn(codes);
    }
}
