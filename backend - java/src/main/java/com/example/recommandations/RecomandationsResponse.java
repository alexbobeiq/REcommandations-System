package com.example.recommandations;
import java.util.List;

public record RecomandationsResponse(Integer customerId, List<String> recommandations) {}
