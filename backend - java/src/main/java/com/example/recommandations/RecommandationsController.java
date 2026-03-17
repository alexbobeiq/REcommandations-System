package com.example.recommandations;

import org.springframework.web.bind.annotation.*;
import java.util.List;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

@RestController
@RequestMapping("/store")
@CrossOrigin(origins = "*")
public class RecommandationsController {

    private static final Logger log = LoggerFactory.getLogger(RecommandationsController.class);
    private final RecommandationsService recommandationsService;

    // CONSTRUCTORUL MANUAL
    public RecommandationsController(RecommandationsService recommandationsService) {
        this.recommandationsService = recommandationsService;
    }

    @GetMapping("/recommandations/{customerID}")
    public List<Product> getRecommandations(@PathVariable Integer customerID) {
        log.info("Cerere pentru clientul: {}", customerID);
        return recommandationsService.getRecommandationsProducts(customerID);
    }
}